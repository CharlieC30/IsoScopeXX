# Copyright 2020 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.model_utils import get_activation

sig = nn.Sigmoid()
ACTIVATION = nn.ReLU
#device = 'cuda'


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.reshape(x.size()[0], -1)

def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))

    return torch.cat((upsampled, bypass), 1)



def get_normalization(out_channels, method):
    if method == 'batch':
        return nn.BatchNorm2d(out_channels)
    elif method == 'instance':
        return nn.InstanceNorm2d(out_channels)
    elif method == 'group':
        return nn.GroupNorm(32, out_channels)
    elif method == 'none':
        return nn.Identity()

def get_normalization_3d(out_channels, method):
    if method == 'batch':
        return nn.BatchNorm3d(out_channels)
    elif method == 'instance':
        return nn.InstanceNorm3d(out_channels)
    elif method == 'group':
        return nn.GroupNorm(32, out_channels)
    elif method == 'none':
        return nn.Identity()


def conv2d_bn_block(in_channels, out_channels, kernel=3, momentum=0.01, activation=ACTIVATION, norm='batch'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=1),
        #nn.BatchNorm2d(out_channels, momentum=momentum),
        get_normalization(out_channels, method=norm),
        activation(),
    )


def deconv2d_bn_block(in_channels, out_channels, use_upsample=True, kernel=4, stride=2, padding=1, momentum=0.01,
                      activation=ACTIVATION, norm='batch'):
    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        up,
        #nn.BatchNorm2d(out_channels, momentum=momentum),
        get_normalization(out_channels, method=norm),
        activation(),
    )


def conv3d_bn_block(in_channels, out_channels, kernel=3, momentum=0.01, activation=ACTIVATION, norm='batch'):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel, padding=1),
        #nn.BatchNorm3d(out_channels, momentum=momentum),
        get_normalization_3d(out_channels, method=norm),
        activation(),
    )


def deconv3d_bn_block(in_channels, out_channels, use_upsample=(2, 2, 2), kernel=4, stride=2, padding=1, momentum=0.01,
                      activation=ACTIVATION, norm='batch'):
    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=use_upsample),
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose3d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        up,
        #nn.BatchNorm3d(out_channels, momentum=momentum),
        get_normalization_3d(out_channels, method=norm),
        activation(),
    )


class Generator(nn.Module):
    def __init__(self, n_channels=1, out_channels=1, nf=32, norm_type='batch', activation=ACTIVATION, final='tanh', mc=False, use_upsample=False):
        super(Generator, self).__init__()

        if norm_type == 'batch':
            batch_norm = True

        conv2_block = conv2d_bn_block
        conv3_block = conv3d_bn_block

        max2_pool = nn.MaxPool2d(2)
        self.max3_pool = nn.MaxPool3d(2)
        act = activation

        if mc:
            dropout = 0.5
        else:
            dropout = 0.0

        self.down0 = nn.Sequential(
            conv2_block(n_channels, nf, activation=act, norm='none'),
            conv2_block(nf, nf, activation=act, norm=norm_type)
        )
        self.down1 = nn.Sequential(
            conv2_block(nf, 2 * nf, activation=act, norm=norm_type),
            conv2_block(2 * nf, 2 * nf, activation=act, norm=norm_type),
        )
        self.down2 = nn.Sequential(
            conv2_block(2 * nf, 4 * nf, activation=act, norm=norm_type),
            nn.Dropout(p=dropout, inplace=False),
            conv2_block(4 * nf, 4 * nf, activation=act, norm=norm_type),

        )
        self.down3 = nn.Sequential(
            conv2_block(4 * nf, 8 * nf, activation=act, norm=norm_type),
            nn.Dropout(p=dropout, inplace=False),
            conv2_block(8 * nf, 8 * nf, activation=act, norm=norm_type),
        )

        self.up3 = deconv3d_bn_block(8 * nf, 4 * nf, activation=act, norm=norm_type, use_upsample=use_upsample)

        self.conv5 = nn.Sequential(
            conv3_block(4 * nf, 4 * nf, activation=act, norm=norm_type),  # 8
            nn.Dropout(p=dropout, inplace=False),
            conv3_block(4 * nf, 4 * nf, activation=act, norm=norm_type),
        )
        self.up2 = deconv3d_bn_block(4 * nf, 2 * nf, activation=act, norm=norm_type, use_upsample=use_upsample)
        self.conv6 = nn.Sequential(
            conv3_block(2 * nf, 2 * nf, activation=act, norm=norm_type),
            nn.Dropout(p=dropout, inplace=False),
            conv3_block(2 * nf, 2 * nf, activation=act, norm=norm_type),
        )

        self.up1 = deconv3d_bn_block(2 * nf, nf, activation=act, norm=norm_type, use_upsample=use_upsample)

        final_layer = get_activation(final)

        self.conv7_k = nn.Sequential(
            conv3_block(nf, out_channels, activation=final_layer, norm='none'),
        )

        self.conv7_g = nn.Sequential(
            conv3_block(nf, out_channels, activation=final_layer, norm='none'),
        )

        self.encoder = nn.Sequential(self.down0, self.down1, self.down2, self.down3)
        self.decoder = nn.Sequential(self.up3, self.conv5, self.up2, self.conv6, self.up1)

    def forward(self, x, method=None):
        # x: (B, C, H, W, D) -- B preserved throughout (for method='decode' directly)
        if method != 'decode':
            # Non-decode path: flatten B*D for 2D encoder, B preserved via reshape
            B, C, H, W, D = x.shape
            x = x.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)
            # (B*D, C, H, W) -- D-slices flattened into 2D batch dim
            feat = []
            for i in range(len(self.encoder)):
                if i > 0:
                    # reshape back to 5D for MaxPool3d (which pools H, W, D)
                    BD, C_cur, H_cur, W_cur = x.shape
                    D_cur = BD // B
                    x = x.reshape(B, D_cur, C_cur, H_cur, W_cur).permute(0, 2, 3, 4, 1)
                    # (B, C, H_cur, W_cur, D_cur)
                    x = self.max3_pool(x)
                    # pools all 3 spatial dims by 2: (B, C, H/2, W/2, D/2)
                    _, C_cur, H_cur, W_cur, D_cur = x.shape
                    x = x.permute(0, 4, 1, 2, 3).reshape(B * D_cur, C_cur, H_cur, W_cur)
                    # back to 4D for 2D conv
                x = self.encoder[i](x)
                BD_f, C_f, H_f, W_f = x.shape
                D_f = BD_f // B
                feat.append(x.reshape(B, D_f, C_f, H_f, W_f).permute(0, 2, 3, 4, 1))
                # feat[i]: (B, C, H_i, W_i, D_i) -- skip connection as 5D with B preserved
            if method == 'encode':
                return feat
            BD_f, C_f, H_f, W_f = x.shape
            D_f = BD_f // B
            x = x.reshape(B, D_f, C_f, H_f, W_f).permute(0, 2, 3, 4, 1)
            # (B, C, H_enc, W_enc, D_enc) -- final encoder output as 5D with B preserved
        # decoder input: (B, C, H_in, W_in, D_in) -- 3D deconv with 3 levels of 2x upsample
        x = self.decoder(x)
        x70 = self.conv7_k(x)
        x71 = self.conv7_g(x)
        # output: (B, 1, H, W, D) -- two output heads

        return {'out0': x70, 'out1': x71}


if __name__ == '__main__':
    g = Generator(n_channels=1, norm_type='group', final='tanh', use_upsample=False)
    #from torchsummary import summary
    #from utils.data_utils import print_num_of_parameters
    #print_num_of_parameters(g)
    f = g(torch.rand(1, 1, 128, 128, 128), method='encode')
    print(f[-1].shape)
    #f = g(torch.rand(1, 1, 128, 128, 16), method='encode')
    #print(f[-1].shape)
    #upsample = torch.nn.Upsample(size=(16, 16, 16))
    #fup = upsample(f[-1])
    #print(fup.shape)
    out = g(f[-1], method='decode')
    print(out['out0'].shape)
