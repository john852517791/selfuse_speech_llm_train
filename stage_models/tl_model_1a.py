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

class base_model(L.LightningModule):
    def __init__(self, 
                 conf_yaml,
                 ) -> None:
        super().__init__()
        self.conf = toolcfg.dict_to_namespace(conf_yaml)
        self.save_hyperparameters(self.conf)
        self.loss_criterion = CTCLoss(blank=151646)
        self.encoder = speechEncoder(
            self.conf.fbank_conf.num_mel_bins, 
            **conf_yaml['encoder_conf'])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.conf.basic_settings.llm_path,
            trust_remote_code=True
            )
        self.ctc_head = nn.Linear(
            self.conf.encoder_conf.overview_conf.encoder_output_dim, 
            self.tokenizer.vocab_size+4, # vocab_size + 3 special tokens + 1 blank
            bias=False)
        # model_conf = conf_yaml["model_conf"]
        # self.adapter = CNNSubsampling(
        #     model_conf["enc_out_dim"], model_conf["llm_embed_dim"], 
        #     model_conf["kernel_size"], model_conf["activation_func"], 
        #     model_conf["norm"])
        
    def forward(self,batch):
        fbank_batch_padded, fbank_len_batch, _, _, _ = batch
        encoder_out,encoder_mask = self.encoder(fbank_batch_padded, fbank_len_batch)
        # inputs_embeds, encoder_mask, _ = self.adapter(
        #     encoder_out, encoder_mask, 
        #     cache=None, return_cache=False) # 1, T, D
        input_lengths = torch.tensor([torch.sum(ele).item() for ele in encoder_mask],dtype=torch.long).to(encoder_out.device)
        inputs_embeds = self.ctc_head(encoder_out)
        inputs_embeds = F.log_softmax(inputs_embeds, dim=-1)
        return inputs_embeds,input_lengths
    
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
        inputs_embeds,input_lengths = self.forward(batch)
        bs = inputs_embeds.shape[0]
        eval_loss = self.loss_criterion(
            inputs_embeds.transpose(0,1),
            batch[2],
            input_lengths,
            batch[3],
            )
        res,codes = ctc_greedy_decode(inputs_embeds,self.tokenizer)
        # # Logging to TensorBoard (if installed) by default
        self.log_dict({
            "eval_loss": eval_loss,
            },on_step=True, 
                # on_epoch=True,
                prog_bar=True, logger=True,
                # prevent from saving wrong ckp based on the eval_loss from different gpus
                sync_dist=True, 
                )
        tensorboard = self.logger.experiment
        for i in range(batch[0].shape[0]):
            tensorboard.add_text(
                f'transcription_{i}', res[i], self.global_step
            )
            tensorboard.add_text(
                f'ground_truth_{i}', batch[4][i], self.global_step
            )
            tensorboard.add_text(
                f'codes_{i}', str(codes[i]), self.global_step
            )
            tensorboard.add_text(
                f'gt_codes_{i}', str(batch[2]), self.global_step
            )

    
    def test_step(self, batch,) -> Any:
        inputs_embeds,_ = self.forward(batch)
        res,codes = ctc_greedy_decode(inputs_embeds,self.tokenizer)
        print(res)
        return 
    
    def predict_step(self, batch, batch_idx):
        inputs_embeds,_ = self.forward(batch)
        res,codes = ctc_greedy_decode(inputs_embeds,self.tokenizer)
        print(res)
        return 

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