from typing import Any
import lightning as L
import torch
from torch.nn import CTCLoss
import utils.tools.config as toolcfg
import logging,os
from models.encoder.encoder import speechEncoder
from models.adapter import *
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from utils.load_data.load_asr_data import ctc_greedy_decode
from transformers import AutoModelForCausalLM

class base_model(L.LightningModule):
    def __init__(self, 
                 conf_yaml,
                 ) -> None:
        super().__init__()
        self.conf = toolcfg.dict_to_namespace(conf_yaml)
        self.save_hyperparameters(self.conf)

        # tokenizer of LLM
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.conf.basic_settings.llm_path,
            trust_remote_code=True
            )
        # embedding layer of LLM
        self.LLM_embedding_layer = AutoModelForCausalLM.from_pretrained(
            self.conf.basic_settings.llm_path, 
            torch_dtype=torch.float32,
            trust_remote_code=True).model.embed_tokens
        # NAR speech decoder

        # AR speech decoder

        # ticodec encoder
        




    def forward(self,batch):
        fbank_batch_padded, fbank_len_batch, _, _, _ = batch

        return 
    
    def training_step(self, batch, batch_idx):
        inputs_embeds,input_lengths = self.forward(batch)

        batch_loss = self.loss_criterion(
            inputs_embeds.transpose(0,1),
            batch[2],
            input_lengths,
            batch[3],
            )
        
        batch_loss = batch_loss.mean()
        self.log_dict({
            "trn_loss": batch_loss,
            },on_step=True, 
                # on_epoch=True,
                prog_bar=True, logger=True,
                # prevent from saving wrong ckp based on the eval_loss from different gpus
                sync_dist=True, 
                )
        return batch_loss
        
    def validation_step(self,batch):
        pass
    def test_step(self, batch,) -> Any:
        pass    
    def predict_step(self, batch, batch_idx):
        pass
    def configure_optimizers(self):
        lr = (self.conf.basic_settings.lr)
        weight_decay = (self.conf.basic_settings.weight_decay)
        optimizer_params = [
            {'params': self.encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            {'params': self.ctc_head.parameters(), 'lr': lr, 'weight_decay': weight_decay}
        ]
        configure = {
            "optimizer": torch.optim.AdamW(optimizer_params),
            }
            
        return configure