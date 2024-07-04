
import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
import pandas as pd
import torch
from lightning import LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from anfsd.classification.training import get_pretrained_model_checkpoint

TARGET_CLASSES = ["noise", "siren"]
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def inference(config: DictConfig) -> pd.DataFrame:
    """Perform inference with a trained model."""

    log.info(f"Inference configuration\n{OmegaConf.to_yaml(config, resolve=True)}")

    inference_config = config.inference

    # Output path
    output_path = Path(config.inference.output_path).with_suffix(".csv")
    if output_path.exists() and not inference_config.get("overwrite", False):
        raise FileExistsError(f"Output file already exists at: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load inference index to make sure it exists
    df_predictions = pd.read_csv(inference_config.dataset.index)[["file_path"]]

    # Load model from checkpoint
    ckpt_path = get_pretrained_model_checkpoint(
        pretrained_model=inference_config.alias,
        work_folder=config.env.work_folder,
    )
    model: LightningModule = hydra.utils.get_class(config.model._target_).load_from_checkpoint(ckpt_path, **config.model)

    # Load prediction dataset
    inference_dataset: Dataset = instantiate(inference_config.dataset)

    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=inference_config.batch_size,
        shuffle=False,
        persistent_workers=False,
        num_workers=inference_config.num_workers,
    )

    log.info(f"Inference dataset: {len(inference_dataset)} segments")

    # Predict
    trainer = Trainer(logger=False, accelerator=inference_config.accelerator)
    predictions = trainer.predict(
        model=model,
        dataloaders=inference_dataloader,
        return_predictions=True,
    )

    # Store predictions
    df_predictions["prediction"] = torch.cat([batch for batch in predictions])

    # Store results
    df_predictions.to_csv(output_path, index=False)
    log.info(f"Predictions stored at: {output_path}")

    return df_predictions


if __name__ == "__main__":
    inference()