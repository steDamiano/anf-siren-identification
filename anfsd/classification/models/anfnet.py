import torch
from torch import nn, Tensor
import torch.nn.functional as F
from lightning import LightningModule
import torchmetrics

class ANFNet(LightningModule):
    def __init__(
            self,
            *,
            n_filt: int = 16,
            kernel_size: int = 32,
            num_channels: int = 1,
            optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
            learning_rate: float = 0.001
    ):
        super().__init__()
        self.n_filt = n_filt
        self.kernel_size = kernel_size
        self.num_channels = num_channels

        if self.num_channels == 1:
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.n_filt, kernel_size=self.kernel_size, stride=2)
        elif self.num_channels == 2:
            self.conv1 = nn.Conv1d(in_channels=2, out_channels=self.n_filt, kernel_size=self.kernel_size, stride=2)

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=self.n_filt, out_channels=self.n_filt * 2, kernel_size=self.kernel_size // 2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=self.n_filt * 2, out_channels= self.n_filt * 4, kernel_size=self.kernel_size // 4, stride=2)
        
        self.fc1 = nn.Linear(in_features=self.n_filt * 4, out_features=self.n_filt * 4)
        self.fc2 = nn.Linear(in_features=self.n_filt * 4, out_features=self.n_filt * 2)
        self.fc3 = nn.Linear(in_features=self.n_filt * 2, out_features=1)

        self.metrics = {
            "train": {
                "accuracy": torchmetrics.Accuracy(task='binary')
            },
            "val": {
                "accuracy": torchmetrics.Accuracy(task='binary')
            }

        }

        # Register metrics to allow automatic device placement
        for split in self.metrics:
            for label, metric in self.metrics[split].items():
                self.register_module(f"metric_{split}_{label}", metric)

        self.optimizer_partial = optimizer
        self.learning_rate = learning_rate

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.25)

        output = F.sigmoid(self.fc3(x))

        return output.squeeze()

    def training_step(self, batch: dict[str, Tensor], batch_idx: int):
        output = self(batch["audio"])
        self._update_metrics(split="train", output=output, batch=batch)
        return self._compute_loss(split="train", output=output, batch=batch)
    
    def on_train_epoch_end(self) -> None:
        """Hook at the end of the training epoch."""
        self._log_metrics(split="train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        output = self(batch["audio"])
        self._update_metrics(split="val", output=output, batch=batch)
        return self._compute_loss(split="val", output=output, batch=batch)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics(split="val")

    def predict_step(self, batch: dict[str, Tensor], batch_idx = int):
        output = self(batch["audio"])
        return output
        
    def _compute_loss(self, split, output, batch):
        batch_size = batch["audio"].shape[0]
        loss = F.binary_cross_entropy(output, batch["label"].float())

        self.log(
            f"{split}/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size
        )
        
        return loss
    
    def _update_metrics(self, split: str, output: dict[str, Tensor], batch: dict[str, Tensor]) -> None:
        split_metrics = self.metrics[split]

        for metric in split_metrics.values():
            metric.update(output, batch["label"].float())
    
    def _log_metrics(self, *, split: str) -> None:
        """Log the metrics for the given split."""
        split_metrics = self.metrics[split]
        for metric_name, metric in split_metrics.items():
                self.log(
                    f"{split}/{metric_name}",
                    metric.compute(),
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                )
                metric.reset()
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for the baseline model."""
        return self.optimizer_partial(self.parameters(), lr=self.learning_rate)