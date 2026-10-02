import os
from huggingface_hub import hf_hub_download

IMAGENET_REPO = "nvidia/PixelDiT-ImageNet"
T2I_REPO = "nvidia/PixelDiT-1300M-1024px"
IMAGENET_V2_REPO = "nvidia/PixelDiT2-ImageNet"

KNOWN = {
    "imagenet256_pixeldit_xl_epoch80.ckpt": IMAGENET_REPO,
    "imagenet256_pixeldit_xl_epoch160.ckpt": IMAGENET_REPO,
    "imagenet256_pixeldit_xl_epoch320.ckpt": IMAGENET_REPO,
    "imagenet512_pixeldit_xl.ckpt": IMAGENET_REPO,
    "pixeldit_t2i_v1.pth": T2I_REPO,
    "pixeldit2_h16_in256_ep600.ckpt": IMAGENET_V2_REPO,
    "pixeldit2_h16_in512_ep680.ckpt": IMAGENET_V2_REPO,
}


def resolve_checkpoint(path):
    if not path or os.path.exists(path):
        return path
    name = os.path.basename(path)
    if name in KNOWN:
        return hf_hub_download(KNOWN[name], name)
    return path
