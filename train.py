import lightning as L
import argparse
import importlib
from lightning.pytorch import loggers as pl_loggers
import utils.b_tools.config as toolcfg
from pytorch_lightning import seed_everything
import torch

def main(args):
    # arguments initialization
    conf = toolcfg.yaml2namespace(args.conf)
    basicSet = conf.basic_settings

    # seed
    seed_everything(seed=basicSet.loss,workers = True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # config the base model containing train eval test and inference funtion
    tl_model = importlib.import_module(toolcfg.convert_path(basicSet.tl_model))

    # config the data module containing the train set, dev set and test set
    dm_module = importlib.import_module(toolcfg.convert_path(basicSet.data_module))
    dm_here = dm_module.data_module(basicSet)

    # model init
    prj_model = importlib.import_module(toolcfg.convert_path(args.module_model))
    model = prj_model.Model(conf)

    # init model, including loss func, optim and scheduler 
    customed_model_wrapper = tl_model.base_model(
            model=model,
            args=conf
            )

    # config logdir
    tb_logger = pl_loggers.TensorBoardLogger(basicSet.savedir, name="")

    # get callbacks
    cbs = getattr(conf, 'callbacks', None)
    call_backs = toolcfg.get_callbacks(conf.callbacks) if cbs!=None else []

    # model initialization
    trainer = L.Trainer(
        accelerator = "gpu",
        devices = basicSet.gpuid,
        max_epochs=basicSet.epochs,
        strategy='ddp_find_unused_parameters_true',
        log_every_n_steps = basicSet.log_every_n_steps,
        callbacks=call_backs,
        check_val_every_n_epoch=basicSet.check_val_every_n_epoch,
        logger=tb_logger,
        enable_progress_bar=False
        )
    trainer.fit(
        model=customed_model_wrapper, 
        datamodule=dm_here
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='General argument parse'
        )
    mes=""
    # random seed
    parser.add_argument('--config', type=str, default="config/stage_1_sample.yaml")
    args = parser.parse_args()
    main(args)  

