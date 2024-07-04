import torch
from torch import nn, Tensor
import torch.nn.functional as F
from lightning import LightningModule
import torchmetrics

class VGGSiren(LightningModule):
    def __init__(
            self,
            num_filters = [4, 8, 16],
            optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
            learning_rate: float = 0.001,
            **kwargs
    ):
        super(VGGSiren, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=num_filters[0], kernel_size=(3,3), padding=(1,1), bias=True)
        self.conv1_2 = nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[0], kernel_size=(3,3), padding=(1,1), bias=True)

        self.conv2_1 = nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=(3,3), padding=(1,1), bias=True)
        self.conv2_2 = nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[1], kernel_size=(3,3), padding=(1,1), bias=True)

        self.conv3_1 = nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=(3,3), padding=(1,1), bias=True)
        self.conv3_2 = nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[2], kernel_size=(3,3), padding=(1,1), bias=True)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features = 3584, out_features=10)
        self.output = nn.Linear(in_features=10, out_features=1)

        self.metrics = {
            "train": {
                "accuracy": torchmetrics.Accuracy(task='binary')
            },
            "val": {
                "accuracy": torchmetrics.Accuracy(task='binary')
            },
            "test": {
                "accuracy": torchmetrics.Accuracy(task='binary')
            }

        }
        self.test_outputs = []
        self.test_labels = []

        # Register metrics to allow automatic device placement
        for split in self.metrics:
            for label, metric in self.metrics[split].items():
                self.register_module(f"metric_{split}_{label}", metric)

        self.optimizer_partial = optimizer
        self.learning_rate = learning_rate

    def forward(self, input: torch.Tensor):
        x = F.elu(self.conv1_1(input))
        x = F.max_pool2d(F.elu(self.conv1_2(x)), kernel_size=(2,2), stride=(2,2))

        x = F.elu(self.conv2_1(x))
        x = F.max_pool2d(F.elu(self.conv2_2(x)), kernel_size=(2,2), stride=(2,2))

        x = F.elu(self.conv3_1(x))
        x = F.max_pool2d(F.elu(self.conv3_2(x)), kernel_size=(2,2), stride=(2,2))
        
        x = self.flatten(x)

        x = F.dropout(x, p=0.1)

        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=0.1)
        
        x = F.sigmoid(self.output(x))

        return x.squeeze()
    
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
    
    def test_step(self, batch: dict[str, Tensor], batch_idx = int):
        output = self(batch["audio"])
        self.test_outputs.append(output)
        self.test_labels.append(batch["label"])
        self._update_metrics(split="test", output=output, batch=batch)
        self._log_metrics(split="test")

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