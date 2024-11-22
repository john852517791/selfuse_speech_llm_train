import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch import Tensor
import sys
sys.path.append("./")
from models.components.speech_encoder import encoder_type
from models.components.speech_projector import EncoderProjectorConcat
from models.components.llm import get_llm
import pytorch_lightning as pl
from torch.optim import Adam



IGNORE_INDEX = -100
SPEECH_TOKEN_INDEX = -200
DEFAULT_SPEECH_TOKEN = "<speech>"


class SpeechLLMLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder_type[args.encoder_type](args.encoder_dir)
        self.tokenizer, self.llm_model = get_llm(
            self.args.llm_dir,
            self.args.use_lora,
            self.args.lora_alpha 
            )
        # freeze whisper_encoder
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False
        
    def training_step(self, batch, batch_idx):
        prompt,speech_tensor,speech_length,answer_tokens = batch
        outputs = self.forward(prompt,speech_tensor,speech_length,answer_tokens)
        loss =  outputs["loss"]
        self.log("train/loss", loss, on_epoch=False)
        return loss
    
    def configure_optimizers(self):
        opt = [
            {"params": self.whisper_encoder.parameters(), "lr": 1e-5},
            {"params": self.speech_projector.parameters(), "lr": self.max_lr},
            {"params": self.llm_model.parameters(), "lr": self.max_lr},
        ]
        optimizer = Adam(opt, lr=self.max_lr)
        return optimizer

    def forward(self, prompt,speech_tensor,speech_length,answer_tokens):
        # input: text token,audio tensor,audio length 
        combined_embeds, atts, label_ids = self.encode(
            prompt,
            speech_tensor,
            speech_length,
            answer_tokens
            )
        out = self.llm_model(
            inputs_embeds=combined_embeds,
            attention_mask=atts,
            labels=label_ids,
        )
        return out

    def encode(self,prompt,speech_tensor,speech_length,answer_tokens):
        batch_size = speech_tensor.shape[0]

        prompts = prompt.split("<speech>")
        front_prompt,back_promt = prompts[0],prompts[1]
        front_tokens, back_tokens = self.tokenizer(front_prompt), self.tokenizer(back_promt)
        # whisper encoder, adaptor
        speech_features = self.encoder.prepare_speech(speech_tensor,speech_length)
        combined_embeds = torch.cat([front_tokens,speech_features,back_tokens,answer_tokens],dim=1)
        atts = torch.ones(combined_embeds.size()[:-1]).to(combined_embeds.device)

        input_token_length = front_tokens.shape[1] + speech_features.shape[1] + back_tokens.shape[1]
        label_ids = torch.cat([
            torch.ones([batch_size, input_token_length], device=combined_embeds.device)*-100,
            answer_tokens
        ], 1).to(combined_embeds.device)
        return combined_embeds, atts, label_ids

        # llama

    




    
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    md = Model(None)
    # print(summary(md, torch.randn((8,64600)), show_input=False))
    op,res = md( torch.randn((8,64600)))
    print(op.shape)