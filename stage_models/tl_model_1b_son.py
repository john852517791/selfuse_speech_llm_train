from typing import Any
import lightning as L
import torch
from torch.nn import CTCLoss
import utils.tools.config as toolcfg
import logging,os
from models.encoder.encoder import speechEncoder,make_pad_mask
from models.adapter import *
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import torch.nn.functional as F
from utils.load_data.load_asr_data import ctc_greedy_decode,post_decode
from stage_models.tl_model_1a import base_model as stage_1a

class base_model(stage_1a):
    def __init__(self, 
                 conf_yaml,
                 ) -> None:
        super().__init__(conf_yaml)
        adaptor_conf = conf_yaml["adapter_conf"]
        self.adapter = CNNSubsampling(
            adaptor_conf["enc_out_dim"], adaptor_conf["llm_embed_dim"], 
            adaptor_conf["kernel_size"], adaptor_conf["activation_func"], 
            adaptor_conf["norm"])
        
        self.LLM = AutoModelForCausalLM.from_pretrained(self.conf.basic_settings.llm_path, 
                                                    torch_dtype=torch.float32,
                                                    trust_remote_code=True)
        for param in self.LLM.parameters():
            param.requires_grad = False

        # define task ids
        special_tk_conf = conf_yaml["special_tk_conf"]
        self.task_embeddings = torch.nn.Embedding(special_tk_conf["task_num"], adaptor_conf["llm_embed_dim"])
        self.task_ids = {
            "sot": 0,
            "transcribe": 1,
            "translate": 2,
            "zh": 3,
            "en": 4,
            "audio": 5,
            "/audio": 6,
            "hyps": 7,
            "/hyps": 8,
        }
        self.pretend = torch.tensor([
            self.task_ids["translate"],
            self.task_ids["zh"],
            self.task_ids["audio"],
            self.task_ids["/audio"], 
            self.task_ids["sot"]]).long()
        self.label_pretend = self.pretend.unsqueeze(0).expand(self.conf.basic_settings.batch_size, -1)
        
        
    def forward(self,batch):
        fbank, fbank_len, target, target_len,texts = batch
        special_token_emb = self.task_embeddings(self.label_pretend.to(target.device)).to(target.device)

        encoder_out ,encoder_mask = self.encoder(fbank,fbank_len)
        # adapter
        inputs_embeds, encoder_mask, _ = self.adapter(
            encoder_out, encoder_mask, 
            cache=None, return_cache=True)
        target_len_mask = ~make_pad_mask(target_len, target.size(1)).unsqueeze(1)
        # inputs_embeds，encoder_mask，target，target_len_mask
        # 1、获取输出的3584embedding
        outputs_embed = self.LLM.model.embed_tokens(target)
        # 2、将两个3584拼接传入inputs_embeds字段
        all_inputs_embeds = torch.cat([special_token_emb[:,:-2,:],inputs_embeds, special_token_emb[:,-2:,:], outputs_embed], dim=1)
        # 3、拼接attention mask
        pretend_att_mask = torch.ones(self.label_pretend.shape, dtype=torch.bool).unsqueeze(1).to(target.device)
        all_attmask = torch.cat([pretend_att_mask[:,:,:-2], encoder_mask, pretend_att_mask[:,:,-2:], target_len_mask], dim=-1).squeeze(1)
        # 4、构建labels
        labels_inputs = torch.full((inputs_embeds.size(0), inputs_embeds.size(1)), -100, dtype=torch.long).to(target.device)
        labels_pretend = torch.full((self.label_pretend.size(0), self.label_pretend.size(1)), -100, dtype=torch.long).to(target.device)
        labels = torch.cat([labels_pretend[:,:-2],labels_inputs, labels_pretend[:,-2:], target], dim=-1)
        labels[labels == 151643] = -100
        # 5、组成{inputs_embeds，attention map，labels}调用forward
        final_inuts = {
            "inputs_embeds": all_inputs_embeds,
            "attention_mask": all_attmask,
            "labels": labels
        }
        return self.LLM(**final_inuts)

    
    def training_step(self, batch, batch_idx):
        forward_outputs = self.forward(batch)

        batch_loss = forward_outputs.loss
        
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
        forward_outputs = self.forward(batch)

        eval_loss = forward_outputs.loss
        codes = post_decode(forward_outputs.logits)
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
        for i in range(self.conf.basic_settings.batch_size):
            tensorboard.add_text(
                f'transcription_{i}', self.tokenizer.decode(codes[i]), self.global_step
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
        return 
    
    def predict_step(self, batch, batch_idx):
        return 

    def configure_optimizers(self):
        lr = (self.conf.basic_settings.lr)
        weight_decay = (self.conf.basic_settings.weight_decay)
        optimizer_params = [
            {'params': self.encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            {'params': self.adapter.parameters(), 'lr': lr, 'weight_decay': weight_decay}
        ]
        configure = {
            "optimizer": torch.optim.AdamW(optimizer_params),
            }
            
        return configure