from models.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from ldm.modules.diffusionmodules.modelcut import Encoder, Decoder
from ldm.util import instantiate_from_config
import yaml
import numpy as np
from models.CUT import PatchSampleF3D
from networks.networks_cut import Normalize, init_net, PatchNCELoss
from torch.optim.lr_scheduler import LambdaLR
from networks.networks import get_scheduler
import os
from pytorch_msssim import ms_ssim
import tifffile as tiff


class GAN(BaseModel):
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, eval_loader, checkpoints) 

        # VQGAN Model
        # Initialize encoder and decoder
        print('Reading yaml: ' + self.hparams.ldmyaml)
        with open('ldm/' + self.hparams.ldmyaml + '.yaml', "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        ddconfig = config['model']['params']["ddconfig"]

        if self.hparams.tc:
            ddconfig['in_channels'] = 2
            ddconfig['out_ch'] = 1
        self.hparams.netG = self.hparams.netG

        self.hparams.final = 'tanh'
        self.net_g, self.net_d = self.set_networks()

        # VQGAN components
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.embed_dim = config['model']['params']['embed_dim']
        self.n_embed = config['model']['params']['n_embed']

        # Vector Quantizer
        self.quantize = VectorQuantizer(
            self.n_embed,
            self.embed_dim,
            beta=0.25,
            remap=getattr(hparams, 'remap', None),
            sane_index_shape=getattr(hparams, 'sane_index_shape', False)
        )

        # Quantization convolutions
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], self.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)

        # Initialize loss
        self.loss = instantiate_from_config(config['model']['params']["lossconfig"])
        self.discriminator = self.loss.discriminator

        # EMA support
        self.use_ema = getattr(hparams, 'use_ema', False)
        if self.use_ema:
            try:
                from taming.modules.util import LitEma
                self.model_ema = LitEma(self)
                print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
            except ImportError:
                print("LitEma not available, disabling EMA")
                self.use_ema = False

        # Save model names
        self.netg_names = {
            'encoder': 'encoder',
            'decoder': 'decoder',
            'quantize': 'quantize',
            'quant_conv': 'quant_conv',
            'post_quant_conv': 'post_quant_conv',
            'net_g': 'net_g'
        }
        self.netd_names = {'discriminator': 'discriminator', 'net_d': 'net_d'}

        # Configure optimizers
        self.configure_optimizers()

        self.upsample = torch.nn.Upsample(size=(hparams.cropsize, hparams.cropsize, hparams.cropsize), mode='trilinear')
        self.uprate = (hparams.cropsize // hparams.cropz) * hparams.dsp / hparams.usp
        self.uprate = int(self.uprate)
        print('uprate: ' + str(self.uprate))

        # CUT NCE
        if not self.hparams.nocut:
            netF = PatchSampleF3D(
                use_mlp=self.hparams.use_mlp,
                init_type='normal',
                init_gain=0.02,
                gpu_ids=[],
                nc=self.hparams.c_mlp
            )
            self.netF = init_net(netF, init_type='normal', init_gain=0.02, gpu_ids=[])
            feature_shapes = [64, 128, 128, 256]
            self.netF.create_mlp(feature_shapes)

            if self.hparams.fWhich == None:
                self.hparams.fWhich = [1 for i in range(len(feature_shapes))]

            print(self.hparams.fWhich)

            self.criterionNCE = []
            for nce_layer in range(len(feature_shapes)):
                self.criterionNCE.append(PatchNCELoss(opt=hparams))

            self.netg_names['netF'] = 'netF'

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VQGAN")
        parser.add_argument("--skipl1", type=int, default=4)
        parser.add_argument("--tc", action="store_true", default=False)
        parser.add_argument("--l1how", type=str, default='dsp')
        parser.add_argument("--dsp", type=int, default=1, help='extra downsample rate')
        parser.add_argument("--usp", type=float, default=1.0, help='extra upsample rate')
        parser.add_argument("--downbranch", type=int, default=1)
        parser.add_argument("--resizebranch", type=int, default=1)
        parser.add_argument('--lbm_ms_ssim', type=float, default=0, help='weight for ms_ssim loss')
        # VQ specific arguments
        parser.add_argument("--ldmyaml", type=str, default='vqgan')
        parser.add_argument("--use_ema", action='store_true', help='use exponential moving average')
        parser.add_argument("--remap", type=str, default=None, help='remap indices')
        parser.add_argument("--sane_index_shape", action='store_true', help='return indices as bhw')
        parser.add_argument("--lr_g_factor", type=float, default=1.0, help='learning rate factor for generator')
        # CUT specific arguments
        parser.add_argument("--nocut", action='store_true')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--lbNCE', type=float, default=0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--use_mlp', action='store_true')
        parser.add_argument("--c_mlp", dest='c_mlp', type=int, default=256, help='channel of mlp')
        parser.add_argument('--fWhich', nargs='+', help='which layers to have NCE loss', type=int, default=None)
        return parent_parser

    def encode(self, x):
        """Encode input to quantized latent space"""
        h, hbranch, hz = self.encoder(x) #(hz > hbranch (=hz[-1]) > > > > h)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)

        # Process hz for CUT loss (similar to original)
        hz = hz[1::2]  # every other two layer  (Z, C, X, Y)
        hz = [x.permute(1, 2, 3, 0).unsqueeze(0) for x in hz]

        return quant, emb_loss, info, h, hz

    def decode(self, quant):
        """Decode from quantized latent space"""
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input, return_pred_indices=False):
        """Forward pass through VQGAN"""
        quant, diff, (_, _, ind), h, hz = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind, h, hz
        return dec, diff, h, hz, quant

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def adv_loss_six_way(self, x, net_d, truth):
        # x: (B, C, H, W, D) -- flatten 6 orthogonal 2D views across batch (B preserved)
        B, C, H, W, D = x.shape
        view1 = x.permute(0, 2, 1, 4, 3).reshape(B * H, C, D, W)
        print(f'[BATCH-DEBUG] adv_loss_six_way: input={tuple(x.shape)}, view1={tuple(view1.shape)}, truth={truth}')  # TEMP DEBUG - remove after verify
        loss = 0
        # (B*H, C, D, W) -- H slices of DW plane across batch
        loss += self.add_loss_adv(a=view1,
                                  net_d=net_d, truth=truth)
        # (B*H, C, W, D) -- H slices of WD plane across batch
        loss += self.add_loss_adv(a=x.permute(0, 2, 1, 3, 4).reshape(B * H, C, W, D),
                                  net_d=net_d, truth=truth)
        # (B*W, C, D, H) -- W slices of DH plane across batch
        loss += self.add_loss_adv(a=x.permute(0, 3, 1, 4, 2).reshape(B * W, C, D, H),
                                  net_d=net_d, truth=truth)
        # (B*W, C, H, D) -- W slices of HD plane across batch
        loss += self.add_loss_adv(a=x.permute(0, 3, 1, 2, 4).reshape(B * W, C, H, D),
                                  net_d=net_d, truth=truth)
        # (B*D, C, H, W) -- D slices of HW plane across batch
        loss += self.add_loss_adv(a=x.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W),
                                  net_d=net_d, truth=truth)
        # (B*D, C, W, H) -- D slices of WH plane across batch
        loss += self.add_loss_adv(a=x.permute(0, 4, 1, 3, 2).reshape(B * D, C, W, H),
                                  net_d=net_d, truth=truth)
        loss = loss / 6
        return loss

    def get_xy_plane(self, x):
        # x: (B, C, H, W, D) -> (B*D, C, H, W) -- all D slices of HW plane across batch
        B, C, H, W, D = x.shape
        return x.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)

    def generation(self, batch, deterministic=False):
        if self.hparams.cropz > 0:
            if deterministic:
                z_init = 0  # Deterministic for validation
                # print(f"[DEBUG] generation cropz: deterministic=True, z_init={z_init}")
            else:
                z_init = np.random.randint(batch['img'][0].shape[4] - self.hparams.cropz)
                # print(f"[DEBUG] generation cropz: deterministic=False, z_init={z_init}")
            for b in range(len(batch['img'])):
                batch['img'][b] = batch['img'][b][:, :, :, :, z_init:z_init + self.hparams.cropz]

        # extra downsample
        if self.hparams.dsp > 1:
            if deterministic:
                z_init = 0
                # print(f"[DEBUG] generation dsp: deterministic=True, z_init={z_init}")
            else:
                z_init = np.random.randint(self.hparams.dsp)
                # print(f"[DEBUG] generation dsp: deterministic=False, z_init={z_init}")
            for b in range(len(batch['img'])):
                batch['img'][b] = batch['img'][b][:, :, :, :, z_init::self.hparams.dsp]

        if self.hparams.usp != 1:
            for b in range(len(batch['img'])):
                batch['img'][b] = nn.Upsample(scale_factor=(1, 1, self.hparams.usp),  # (B, C, X, Y, Z)
                                              mode='trilinear')(batch['img'][b])

        if self.hparams.tc:
            self.oriX = torch.cat((batch['img'][0], batch['img'][1]), 1)
        else:
            self.oriX = batch['img'][0]  # (B, C, X, Y, Z) # original

        # VQGAN forward pass
        # Flatten B and D so 2D VQGAN processes B*D slices; B is preserved via reshape (not dropped)
        B, C_in, H, W, D_in = self.oriX.shape
        input_slice = self.oriX.permute(0, 4, 1, 2, 3).reshape(B * D_in, C_in, H, W)
        # (B*D, C, H, W) -- D-slices flattened into 2D batch dim
        print(f'[BATCH-DEBUG] input_slice: B={B}, D_in={D_in}, shape={tuple(input_slice.shape)}')  # TEMP DEBUG - remove after verify
        # Make sure input requires gradients if we're training
        if self.training:
            input_slice = input_slice.requires_grad_(True)

        self.reconstructions, self.qloss, _, self.hz, quant = self.forward(input_slice, return_pred_indices=False)
        # reconstructions, quant: (B*D, ..., H/8, W/8)

        if self.hparams.downbranch > 1:
            # (B*D, embed_dim, H/8, W/8) -> (B, embed_dim, H/8, W/8, D) for MaxPool3d on D axis
            BD_q, C_q, H_q, W_q = quant.shape
            D_cur = BD_q // B
            quant = quant.reshape(B, D_cur, C_q, H_q, W_q).permute(0, 2, 3, 4, 1)
            quant = nn.MaxPool3d((1, 1, self.hparams.downbranch))(quant)
            # (B, embed_dim, H/8, W/8, D/downbranch)
            D_cur = quant.shape[-1]
            quant = quant.permute(0, 4, 1, 2, 3).reshape(B * D_cur, C_q, H_q, W_q)
            # back to 4D for VQGAN decoder

        if self.hparams.resizebranch != 1:
            BD_q, C_q, H_q, W_q = quant.shape
            D_cur = BD_q // B
            quant = quant.reshape(B, D_cur, C_q, H_q, W_q).permute(0, 2, 3, 4, 1)
            quant = nn.Upsample(scale_factor=(1, 1, self.hparams.resizebranch), mode='trilinear')(quant)
            D_cur = quant.shape[-1]
            quant = quant.permute(0, 4, 1, 2, 3).reshape(B * D_cur, C_q, H_q, W_q)

        quant = self.decoder.conv_in(quant)
        # (B*D, 256, H/8, W/8) -> (B, 256, H/8, W/8, D) for 3D generator (B preserved)
        BD_f, C_f, H_f, W_f = quant.shape
        D_final = BD_f // B
        quant = quant.reshape(B, D_final, C_f, H_f, W_f).permute(0, 2, 3, 4, 1)
        print(f'[BATCH-DEBUG] net_g input: shape={tuple(quant.shape)}')  # TEMP DEBUG - remove after verify

        self.XupX = self.net_g(quant, method='decode')['out0']
        self.Xup = self.upsample(self.oriX)  # (B, C, H, W, D)
        print(f'[BATCH-DEBUG] XupX: shape={tuple(self.XupX.shape)}, Xup: shape={tuple(self.Xup.shape)}')  # TEMP DEBUG - remove after verify

        #print('reconstructions size', self.reconstructions.shape)
        #print('hbranch size', hbranch.shape)
        #print('XupX size', self.XupX.shape)

        if not self.hparams.nocut:
            self.goutz = self.hz

    def get_projection(self, x, depth, how='mean'):
        if how == 'dsp':
            x = x[:, :, :, :, (self.uprate // 2)::self.uprate * self.hparams.skipl1]
        else:
            x = x.unfold(-1, depth, depth)
            if how == 'mean':
                x = x.mean(dim=-1)
            elif how == 'max':
                x, _ = x.max(dim=-1)
        return x

    def backward_g(self):
        loss_g = 0
        loss_dict = {}

        axx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=True)

        loss_l1 = self.add_loss_l1(a=self.get_projection(self.XupX, depth=self.uprate
                                                                          * self.hparams.skipl1,
                                                         how=self.hparams.l1how),
                                   b=self.oriX[:, :, :, :, ::self.hparams.skipl1])

        # ms_ssim in ZY: flatten B and H for (B*H, C, D, W) 2D comparison
        if self.hparams.lbm_ms_ssim > 0:
            B_ms, C_ms, H_ms, W_ms, D_ms = self.XupX.shape
            loss_ms_ssim = 1 - ms_ssim(self.XupX.permute(0, 2, 1, 4, 3).reshape(B_ms * H_ms, C_ms, D_ms, W_ms),
                                    self.Xup.permute(0, 2, 1, 4, 3).reshape(B_ms * H_ms, C_ms, D_ms, W_ms),
                                    data_range=2.0,
                                    size_average=True,
                                    win_size=7,  # Smaller window
                                    weights=[0.0448, 0.2856, 0.6696])
            loss_dict['ms_ssim'] = loss_ms_ssim
            loss_g += loss_ms_ssim * self.hparams.lbm_ms_ssim # Add weight parameter

        loss_dict['axx'] = axx
        loss_g += axx * self.hparams.adv
        loss_dict['l1'] = loss_l1
        loss_g += loss_l1 * self.hparams.lamb

        # VQGAN loss needs 2D slices; flatten B and D so B is preserved
        B_o, C_o, H_o, W_o, D_o = self.oriX.shape
        oriXpermute = self.oriX.permute(0, 4, 1, 2, 3).reshape(B_o * D_o, C_o, H_o, W_o)
        if self.hparams.tc:
            oriXpermute = self.oriX[:, :1].permute(0, 4, 1, 2, 3).reshape(B_o * D_o, 1, H_o, W_o)
        print(f'[BATCH-DEBUG] backward_g VQGAN loss: oriXpermute={tuple(oriXpermute.shape)}, reconstructions={tuple(self.reconstructions.shape)}')  # TEMP DEBUG - remove after verify

        # VQGAN loss (different from KL autoencoder)
        aeloss, log_dict_ae = self.loss(
            self.qloss,  # VQ loss instead of posterior
            oriXpermute,  # original input
            self.reconstructions,  # reconstructed output
            0,  # optimizer index for generator
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train"
        )
        loss_g += aeloss
        loss_dict['ae'] = aeloss

        # Add VQ commitment loss
        loss_dict['vq'] = self.qloss
        loss_g += self.qloss

        # CUT NCE Loss
        if not self.hparams.nocut:
            # feat q - use stored hz from generation
            feat_q = self.hz

            # feat k - encode generated output; every 8th D slice, flatten (B, D_sub) into batch dim
            _, C_k, H_k, W_k, _ = self.XupX.shape
            input_slice_k = self.XupX.permute(0, 4, 1, 2, 3)[:, 4::8].reshape(-1, C_k, H_k, W_k)
            # (B*D_sampled, C, H, W) -- B preserved
            _, _, _, _, feat_k = self.encode(input_slice_k)

            feat_k_pool, sample_ids = self.netF(feat_k, self.hparams.num_patches, None)
            feat_q_pool, _ = self.netF(feat_q, self.hparams.num_patches, sample_ids)

            total_nce_loss = 0.0
            for f_q, f_k, crit, f_w in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.hparams.fWhich):
                loss = crit(f_q, f_k) * f_w
                total_nce_loss += loss.mean()
            loss_nce = total_nce_loss / 4
            loss_dict['nce'] = loss_nce
            loss_g += loss_nce * self.hparams.lbNCE

        loss_dict['sum'] = loss_g
        return loss_dict

    def backward_d(self):
        loss_d = 0
        loss_dict = {}

        dxx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=False)
        dx = self.add_loss_adv(a=self.get_xy_plane(self.oriX), net_d=self.net_d, truth=True)

        loss_dict['dxx_x'] = dxx + dx
        loss_d += (dxx + dx) * self.hparams.adv

        # VQGAN disc loss needs 2D slices; flatten B and D so B is preserved
        B_o, C_o, H_o, W_o, D_o = self.oriX.shape
        oriXpermute = self.oriX.permute(0, 4, 1, 2, 3).reshape(B_o * D_o, C_o, H_o, W_o)
        if self.hparams.tc:
            oriXpermute = self.oriX[:, :1].permute(0, 4, 1, 2, 3).reshape(B_o * D_o, 1, H_o, W_o)

        # VQGAN discriminator loss
        discloss, log_dict_disc = self.loss(
            self.qloss,  # VQ loss
            oriXpermute,
            self.reconstructions,
            1,  # optimizer index for discriminator
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train"
        )
        loss_d += discloss
        loss_dict['disc'] = discloss

        loss_dict['sum'] = loss_d
        return loss_dict

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def configure_optimizers(self):
        lr_d = self.hparams.lr
        lr_g = getattr(self.hparams, 'lr_g_factor', 1.0) * self.hparams.lr
        print("lr_d", lr_d)
        print("lr_g", lr_g)

        # Generator optimizer includes VQ parameters
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quantize.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()) +
            list(self.net_g.parameters()),
            lr=lr_g, betas=(0.5, 0.9)
        )

        # Discriminator optimizer
        opt_disc = torch.optim.Adam(
            list(self.loss.discriminator.parameters()) + list(self.net_d.parameters()),
            lr=lr_d, betas=(0.5, 0.9)
        )

        # Add scheduler support if specified
        if hasattr(self.hparams, 'scheduler_config') and self.hparams.scheduler_config is not None:
            scheduler = instantiate_from_config(self.hparams.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]

            self.net_g_scheduler = scheduler[0]['scheduler']
            self.net_d_scheduler = scheduler[1]['scheduler']

            return [opt_ae, opt_disc], scheduler

        self.net_g_scheduler = get_scheduler(opt_ae, self.hparams)
        self.net_d_scheduler = get_scheduler(opt_disc, self.hparams)

        return [opt_ae, opt_disc], []

    def save_checkpoint(self, filepath):
        # Combine all the state dicts into a single state dict
        state_dict = {}

        # Encoder
        for k, v in self.encoder.state_dict().items():
            state_dict[f'encoder.{k}'] = v

        # Decoder
        for k, v in self.decoder.state_dict().items():
            state_dict[f'decoder.{k}'] = v

        # Quantizer
        for k, v in self.quantize.state_dict().items():
            state_dict[f'quantize.{k}'] = v

        # Quant Conv
        for k, v in self.quant_conv.state_dict().items():
            state_dict[f'quant_conv.{k}'] = v

        # Post Quant Conv
        for k, v in self.post_quant_conv.state_dict().items():
            state_dict[f'post_quant_conv.{k}'] = v

        # Discriminator
        for k, v in self.discriminator.state_dict().items():
            state_dict[f'loss.discriminator.{k}'] = v

        # EMA if used
        if self.use_ema:
            for k, v in self.model_ema.state_dict().items():
                state_dict[f'model_ema.{k}'] = v

        # Create the checkpoint dictionary
        checkpoint = {
            "state_dict": state_dict,
            "global_step": self.global_step,
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
        }

        # Save additional hyperparameters if needed
        if hasattr(self, 'hparams'):
            checkpoint['hparams'] = self.hparams

        torch.save(checkpoint, filepath)
        print(f"VQGAN model saved to {filepath}")

    @classmethod
    def load_from_checkpoint(cls, filepath, train_loader=None, eval_loader=None, checkpoints=None):
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        hparams = checkpoint['hparams']

        model = cls(hparams, train_loader, eval_loader, checkpoints)

        # Load state dict
        state_dict = checkpoint['state_dict']
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

        model.global_step = checkpoint['global_step']
        print(f"VQGAN model loaded from {filepath}")
        return model

    def resume_from_checkpoint(self, checkpoint_dir, epoch_to_load):
        """
        Loads the model, optimizers, schedulers, and epoch/step information
        from a checkpoint to resume training.
        Args:
            checkpoint_dir (str): The directory where checkpoints are saved.
                                  This should typically correspond to hparams.checkpoint_path.
            epoch_to_load (int or str): The epoch number of the checkpoint to load.
        Returns:
            int: The epoch number loaded (0-indexed). Training should resume from this epoch + 1.
        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        # Assumes 'os' module is imported at the top of the file.
        # e.g., import os
        filepath = os.path.join(checkpoint_dir, f"model_epoch_{epoch_to_load}.pth")

        if not os.path.exists(filepath):
            print(f"Checkpoint file not found: {filepath}")
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        print(f"Loading checkpoint to resume training from: {filepath}")
        # getattr(self, 'device', 'cpu') provides a fallback if self.device is not set
        checkpoint = torch.load(filepath, map_location=getattr(self, 'device', 'cpu'))

        # Load model state_dict (this includes encoder, decoder, quantizer, etc.)
        missing_keys, unexpected_keys = self.load_state_dict(checkpoint['state_dict'], strict=False)
        if missing_keys:
            print(f"Missing keys when loading model state_dict: {', '.join(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys when loading model state_dict: {', '.join(unexpected_keys)}")

        # Load optimizer states
        if 'optimizer_g_state_dict' in checkpoint and hasattr(self, 'optimizer_g'):
            self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            print("Loaded optimizer_g state.")
        else:
            print("Warning: optimizer_g_state_dict not found in checkpoint or self.optimizer_g not initialized.")

        if 'optimizer_d_state_dict' in checkpoint and hasattr(self, 'optimizer_d'):
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
            print("Loaded optimizer_d state.")
        else:
            print("Warning: optimizer_d_state_dict not found in checkpoint or self.optimizer_d not initialized.")

        # Load scheduler states
        if 'scheduler_g_state_dict' in checkpoint and hasattr(self, 'scheduler_g'):
            self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
            print("Loaded scheduler_g state.")
        else:
            print("Warning: scheduler_g_state_dict not found in checkpoint or self.scheduler_g not initialized.")

        if 'scheduler_d_state_dict' in checkpoint and hasattr(self, 'scheduler_d'):
            self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
            print("Loaded scheduler_d state.")
        else:
            print("Warning: scheduler_d_state_dict not found in checkpoint or self.scheduler_d not initialized.")
            
        # Load EMA model state if applicable
        # self.use_ema is set in __init__ based on hparams.
        # self.model_ema is created in __init__ if self.use_ema is True.
        if self.use_ema:
            if 'ema_state_dict' in checkpoint:
                if hasattr(self, 'model_ema'):
                    self.model_ema.load_state_dict(checkpoint['ema_state_dict'])
                    print("Loaded EMA model state.")
                else:
                    # This implies an inconsistency if use_ema is True but model_ema was not created.
                    print("Warning: EMA is enabled (self.use_ema=True) but self.model_ema attribute not found.")
            else:
                print("Warning: EMA is enabled (self.use_ema=True) but 'ema_state_dict' not found in checkpoint.")

        # Update global step
        self.global_step = checkpoint.get('global_step', 0)
        
        # Update current epoch (epoch completed)
        loaded_epoch = checkpoint.get('epoch', -1) 
        if loaded_epoch == -1:
            print("Warning: 'epoch' key not found in checkpoint. Current epoch not updated.")
        
        # self.current_epoch is used by save_checkpoint to store the completed epoch number.
        if hasattr(self, 'current_epoch'):
             self.current_epoch = loaded_epoch
        else:
            # If BaseModel or a training framework usually manages current_epoch, this might be fine.
            # However, save_checkpoint in this class uses self.current_epoch.
            print("Warning: Model does not have 'current_epoch' attribute. Loaded epoch not stored on model instance.")
        
        print(f"Resumed from checkpoint. Last completed epoch: {loaded_epoch}, Global Step: {self.global_step}")
        
        return loaded_epoch
