import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from torchmetrics import AveragePrecision, F1Score, Precision, Recall, Accuracy

import torch
log = logging.getLogger(__name__)

METRICS = ["Accuracy", "AUPRC", "Precision", "Recall", "F1"]
TARGET_CLASSES = ["noise", "siren"]

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def evaluate(config: DictConfig) -> pd.DataFrame:
    """Evaluate inference output against ground truth and compute metrics."""

    log.info(f"Evaluation configuration\n{OmegaConf.to_yaml(config)}")

    # Output path
    output_path = Path(config.evaluation.output_path).with_suffix(".csv")
    if output_path.exists():
        raise FileExistsError(f"Output file already exists at: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load predictions and ground truth
    df_pred = pd.read_csv(config.inference.output_path)
    df_gt = pd.read_csv(config.inference.dataset.index)

    if len(df_pred) != len(df_gt):
        raise ValueError("Predictions and ground truth must contain the same number of samples.")

    df_gt.sort_values("file_path", inplace=True)
    df_pred.sort_values("file_path", inplace=True)

    gt_scores = [TARGET_CLASSES.index(val) for val in df_gt["file_class"]]
    pred_scores = df_pred["prediction"].values

    gt_scores = torch.tensor(gt_scores)
    pred_scores = torch.tensor(pred_scores)

    # Accuracy score
    accuracy = Accuracy(task="binary")
    acc_score = accuracy(pred_scores, gt_scores)

    # AUPRC score
    auprc = AveragePrecision(task="binary")
    auc_score = auprc(pred_scores, gt_scores)

    # Precision - Recall - F1
    precision = Precision(task="binary")
    prec_score = precision(pred_scores, gt_scores)

    recall = Recall(task="binary")
    rec_score = recall(pred_scores, gt_scores)

    f1 = F1Score(task="binary")
    f1_score = f1(pred_scores, gt_scores)
    
    results = pd.DataFrame([acc_score, auc_score, prec_score, rec_score, f1_score], index=METRICS)
    log.info(results)

    # Store in desired directory
    results.to_csv(output_path, index=True, index_label="Metric")
    log.info(f"Evaluation results stored at: {output_path}")

    return results


if __name__ == "__main__":
    evaluate()