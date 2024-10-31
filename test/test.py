import os
import sys
sys.path.append("/data8/wangzhiyong/project/LLM/llama_omni")
sys.path.append("/data8/wangzhiyong/project/LLM/llama_omni/test")
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from lightning.pytorch import loggers as pl_loggers
from torch.utils.data import DataLoader
from utils.b_tools.config import yaml2namespace,load_yaml_config
# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder,config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = config
        self.save_hyperparameters(self.args)


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

config = load_yaml_config('config/stage_1_sample.yaml')
# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder,config)
tb_logger = pl_loggers.TensorBoardLogger("test/", name="a_log/")

# setup data
dataset = MNIST("test/", download=True, transform=ToTensor())
train_loader = DataLoader(dataset,batch_size=600)


# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(
    accelerator = "gpu",
    devices = config["basic_settings"]["gpuid"],
    limit_train_batches=600,
    max_epochs=1,
    logger=tb_logger,
    )
trainer.fit(model=autoencoder, train_dataloaders=train_loader)