"""CLI trainer for Suspect Recognition."""

import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from .data import LitImClsDataModule
from .model import LitArcEncoder

__all__ = ["cli_main"]

PARSER_KWARGS = dict(
    fit=dict(
        default_config_files=["cfg/reid.yaml"],
    ),
)

# https://lightning.ai/docs/pytorch/stable/common/trainer.html
TRAINER_DEFAULTS = dict(
    # https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html
    callbacks=[
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="val_sil_score", patience=10, mode="max"),
        ModelCheckpoint(
            monitor="val_sil_score",
            mode="max",
            save_top_k=5,
            save_last=True,
            filename="{epoch}-{val_sil_score:.3f}",
        ),
    ],
    default_root_dir="runs",
    log_every_n_steps=1,
    check_val_every_n_epoch=1,
    # Performance options.
    deterministic=False,
    benchmark=True,
    # Glitchy, see https://github.com/Lightning-AI/lightning/issues/5558.
    # precision="16-mixed",
    # Useful for debugging.
    # fast_dev_run=True,
    # detect_anomaly=True,
)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.link_arguments("model.im_size", "data.im_size")
        parser.link_arguments("model.sched_steps", "trainer.max_steps")
        parser.link_arguments("data.nclasses", "model.nclasses", apply_on="instantiate")

    def before_instantiate_classes(self):
        if "subcommand" in self.config:
            cfg = self.config[self.config.subcommand].trainer
        else:
            cfg = self.config.get("trainer", None)
        if cfg is not None and not cfg.deterministic:  # type: ignore
            torch.set_float32_matmul_precision("medium")
            # torch.set_float32_matmul_precision("high")


def cli_main(args=None):
    """CLI Entrypoint.

    See https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html#run-from-python.
    """
    run = args is None
    cli = MyLightningCLI(
        LitArcEncoder,
        LitImClsDataModule,
        seed_everything_default=42,
        args=args,
        parser_kwargs=PARSER_KWARGS,
        trainer_defaults=TRAINER_DEFAULTS,
        run=run,
    )
    return cli


if __name__ == "__main__":
    cli_main()
