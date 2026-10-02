"""Lightning CLI entry point for PixelDiT2 image-to-image editing."""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data import DataModule
from src.edit_lightning import EditLightningModel
from main import ReWriteRootDirCli, ReWriteRootSaveConfigCallback


if __name__ == "__main__":
    from tools.download import resolve_checkpoint

    for i, arg in enumerate(sys.argv):
        if arg.startswith("--ckpt_path="):
            sys.argv[i] = (
                f"--ckpt_path={resolve_checkpoint(arg.split('=', 1)[1])}"
            )

    ReWriteRootDirCli(
        EditLightningModel,
        DataModule,
        auto_configure_optimizers=False,
        save_config_callback=ReWriteRootSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
    )
