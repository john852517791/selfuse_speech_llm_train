import lightning as L
import argparse
import importlib
from lightning.pytorch import loggers as pl_loggers
import utils.tools.config as toolcfg
from lightning.pytorch import seed_everything
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

def main(args):
    # arguments initialization
    conf_yaml = toolcfg.load_yaml_config(args.conf)
    # conf = toolcfg.dict_to_namespace(conf_yaml)
    conf = toolcfg.load_yaml_to_class(args.conf)
    basicSet = conf.basic_settings
    # print(conf)

    # seed
    seed_everything(seed=basicSet.seed,workers = True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # config the base model containing train eval test and inference funtion
    tl_model = importlib.import_module(toolcfg.convert_path(basicSet.tl_model))

    # config the data module containing the train set, dev set and test set
    dm_module = importlib.import_module(toolcfg.convert_path(basicSet.data_module))
    dm_here = dm_module.data_module(conf)

    # init model, including loss func, optim and scheduler 
    customed_model_wrapper = tl_model.base_model(
            conf_yaml,
            )
    
    # config logdir
    tb_logger = pl_loggers.TensorBoardLogger(basicSet.savedir, name="")

    # get callbacks
    # cbs = getattr(conf_yaml, 'callbacks', None)
    # call_backs = toolcfg.get_callbacks(conf_yaml["callbacks"]) if cbs!=None else []
    checkpoint_callback = ModelCheckpoint(
        filename="best-{trn_loss:.2f}-{step:02d}", 
        save_top_k=1, monitor="trn_loss",
        save_weights_only=True)
    lr_Monitor = LearningRateMonitor(
        logging_interval="step",
    )

    # model initialization
    trainer = L.Trainer(
        accelerator = "gpu",
        devices = basicSet.gpuid,
        max_epochs=basicSet.epochs,
        strategy='ddp_find_unused_parameters_true',
        log_every_n_steps = basicSet.log_every_n_steps,
        val_check_interval= basicSet.val_check_interval,
        check_val_every_n_epoch=None,
        callbacks=[checkpoint_callback,lr_Monitor],        
        logger=tb_logger,
        enable_progress_bar=False,
        # resume_from_checkpoint=None
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
    parser.add_argument('--conf', type=str, default="config/stage_1c.yaml")
    args = parser.parse_args()
    main(args)  
