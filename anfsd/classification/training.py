import logging
import os
from pathlib import Path

import coolname
import hydra
from hydra.utils import instantiate
from dotenv import load_dotenv
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, NeptuneLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

def get_pretrained_model_checkpoint(pretrained_model: str | Path, work_folder: str | Path, ckpt_name: str = "best") -> Path:
    """
    Compute pre-trained model checkpoint path.

    Args:
        pretrained_model: Pre-trained model name or path
        work_folder: Work folder.
        ckpt_name: select best or last checkpoint

    Returns:
        path to a pre-trained model checkpoint.
    """
    if Path(pretrained_model).is_file():
        return Path(pretrained_model)
    return Path(work_folder) / "models" / pretrained_model / "checkpoints" / (ckpt_name +".ckpt")


@hydra.main(version_base=None, config_path="../configs", config_name="VGGSiren_base")
def train(config: DictConfig) -> Trainer:
    log.info("Training")
    training_config = config.training
    if training_config.alias is None:
        alias = coolname.generate_slug(2)
        training_config.alias = alias
    else:
        alias = training_config.alias
    
    log.info(f"Training alias: {alias}")

    output_dir = training_config.output_folder
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config_path = Path(output_dir) / "config.yaml"
    if config_path.exists() and not training_config.get("overwrite", False):
        raise FileExistsError(
            f"Configuration file exists at {config.path}. Please set 'training.overwrite=True' to overwrite."
        )
    OmegaConf.save(config, config_path, resolve=True)

    log.info(f"Training configuration\n{OmegaConf.to_yaml(config, resolve=True)}")

    seed_everything(training_config.seed)
    log.info(f"Seed: {training_config.seed}")

    train_dataset: Dataset = instantiate(training_config.train_dataset)
    log.info(f"Train dataset: {len(train_dataset)} segments")

    val_dataset: Dataset = instantiate(training_config.val_dataset)
    log.info(f"Val dataset: {len(val_dataset)} segments")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        persistent_workers=training_config.num_workers > 0,
        num_workers=training_config.num_workers,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        persistent_workers=training_config.num_workers > 0,
        num_workers=training_config.num_workers,
        drop_last=True
    )

    training_config.trainer["log_every_n_steps"] = min(
        training_config.trainer["log_every_n_steps"], len(train_dataloader)
    )
    
    model: LightningModule = hydra.utils.instantiate(config.model, learning_rate=config.training.learning_rate)

    loggers = []

    try:
        proxies = {
            key: os.getenv(key) 
            for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"] 
            if key in os.environ
        }
        neptune_tags = training_config.get("tags", None)
        loggers.append(
            NeptuneLogger(
                proxies=proxies,
                name=alias,
                log_model_checkpoints=False,
                tags=OmegaConf.to_container(neptune_tags) if neptune_tags is not None else None,
            )
        )
        log.info("Neptune is installed. Adding logger.")
    except ModuleNotFoundError:
        pass
    
    if len(loggers) == 0:
        log.info("No loggers found. Adding CSVLogger.")
        loggers.append(CSVLogger(save_dir=output_dir, name=alias))

    hparams = OmegaConf.to_container(config, resolve=True)
    hparams["output_dir"] = output_dir
    for logger in loggers:
        logger.log_hyperparams(hparams)
    
    # Prepare callbacks
    early_stop = EarlyStopping(**training_config.callbacks.get("early_stopping", {}))

    model_checkpoint_params = OmegaConf.to_container(
        training_config.callbacks.get("model_checkpoint", {}), resolve=True
    )
    model_checkpoint_params.setdefault("dirpath", Path(output_dir) / "checkpoints")
    model_checkpoint = ModelCheckpoint(**model_checkpoint_params)

    # Prepare trainer and fit
    trainer = Trainer(
        **training_config.get("trainer", {}),
        logger=loggers,
        callbacks=[model_checkpoint, early_stop],
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        **training_config.get("fit", {}),
    )

    log.info(f"Best model checkpoint at: {model_checkpoint.best_model_path}")

    return trainer

if __name__ == "__main__":
    load_dotenv()
    train()