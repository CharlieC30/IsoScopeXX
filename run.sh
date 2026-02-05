# default 8X enhancement
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj default/max10skip4 --env t09 --nocut
# default 4X enhancement
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj default/max10skip4 --env t09 --nocut --downbranch 2 --cropz 32 # (128 // 4)
# 8X enhancement with contrastive loss lbNCE=1
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj default/max10skip4 --env t09 --lbNCE 1


# filopodia on GHCL01
cd /home/aero/charliechang/projects/IsoScopeXX/
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml filopodia_GHCL01 --prj default/max10skip4 --env GHCL01 --nocut --downbranch 2 --cropz 32
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml filopodia_GHCL01 --prj default/ --env GHCL01 --nocut --downbranch 2 --cropz 32

# filopodia on NTHU_Gary3
cd IsoScopeXX
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml filopodia_NTHU_Gary3 --prj default/max10skip4 --env NTHU_Gary3 --nocut --downbranch 2 --cropz 32
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml filopodia_NTHU_Gary3 --prj default/ --env NTHU_Gary3 --nocut --downbranch 2 --cropz 32