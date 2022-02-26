import pytorch_lightning as pl
import torch
import torch_geometric

from torch_geometric.nn import global_mean_pool, SAGEConv
from torch_geometric.data import LightningDataset
from torch_geometric.datasets import MNISTSuperpixels

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(pl.LightningModule):
    """
    Replicating GraphSage Model described in https://arxiv.org/pdf/2003.00982.pdf using PyTorch Geometric.
    Parameter Description from https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/configs/superpixels_graph_classification_GraphSage_MNIST_100k.json
    Model Description from https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/nets/superpixels_graph_classification/graphsage_net.py

    Data Modifications from https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/superpixels.py
    """

    def __init__(
        self,
        n_target_classes: int,
        learning_rate: float = 0.01,
        hidden_dimensions: int = 90,
        linear_dimensions: int = 90,
    ):
        super().__init__()

        self.gcn1 = torch_geometric.nn.Sequential(
            "x, edge_index, batch",
            [
                (torch.nn.Linear(3, hidden_dimensions), "x -> x"),
                (SAGEConv(hidden_dimensions, hidden_dimensions), "x, edge_index -> x"),
                (torch.nn.BatchNorm1d(hidden_dimensions), "x -> x"),
                torch.nn.ReLU(inplace=True),
                (SAGEConv(hidden_dimensions, hidden_dimensions), "x, edge_index -> x"),
                (torch.nn.BatchNorm1d(hidden_dimensions), "x -> x"),
                torch.nn.ReLU(inplace=True),
                (SAGEConv(hidden_dimensions, hidden_dimensions), "x, edge_index -> x"),
                (torch.nn.BatchNorm1d(hidden_dimensions), "x -> x"),
                torch.nn.ReLU(inplace=True),
                (SAGEConv(hidden_dimensions, hidden_dimensions), "x, edge_index -> x"),
                (torch.nn.BatchNorm1d(hidden_dimensions), "x -> x"),
                torch.nn.ReLU(inplace=True),
                (global_mean_pool, "x, batch -> x"),
                torch.nn.Linear(linear_dimensions, linear_dimensions // 2),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(linear_dimensions // 2, linear_dimensions // 4),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(linear_dimensions // 4, n_target_classes),
            ],
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x, edge_index, batch):
        return self.gcn1(x, edge_index, batch)

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion(y_hat, batch.y)
        self.log("lr", self.learning_rate, prog_bar=True)
        self.log("loss", loss)
        self.log(
            "acc",
            (torch.sum(torch.argmax(y_hat, 1) == batch.y) / len(batch.y)).item() * 100,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion(y_hat, batch.y)
        self.log(
            "val_acc",
            (torch.sum(torch.argmax(y_hat, 1) == batch.y) / len(batch.y)).item() * 100,
            prog_bar=True,
        )
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Patience can be 5 or 10
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss"}


def train():
    batch_size = 128

    def f(x):
        # https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/superpixels.py#L109-L124
        x.x = torch.cat([x.x, x.pos / 28], 1)
        return x

    datamodule = LightningDataset(
        train_dataset=MNISTSuperpixels(".", transform=f),
        val_dataset=MNISTSuperpixels(".", train=False, transform=f),
        batch_size=batch_size,
        num_workers=0,
    )
    model = GCN(n_target_classes=10, learning_rate=1e-4).to(DEVICE)
    torch.multiprocessing.set_start_method("spawn")
    trainer = pl.Trainer(gpus=1, accumulate_grad_batches=1, max_epochs=500)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
