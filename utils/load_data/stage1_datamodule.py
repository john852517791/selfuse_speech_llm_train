import sys
sys.path.append("./")
from utils.b_tools.prompt import conv_llama_3, tokenizer_speech_token
import numpy as np
import soundfile as sf
import torch,os
from torch import Tensor
from torch.utils.data import Dataset,DataLoader,DistributedSampler
import lightning as L
import whisper
import json
        
class asvspoof_dataModule(L.LightningDataModule):
        def __init__(self,args):
                super().__init__()
                self.args = args
                

        def setup(self, stage: str):
            # Assign train/val datasets for use in dataloaders
            if stage == "fit":
                # pass
                self.trn_set = CustomDataset(self.tokenizer,self.args.train_jsonfile)
   
            # Assign test dataset for use in dataloader(s)
            if stage == "test":
                pass

            if stage == "predict":
                pass
                    
                    
                

        def train_dataloader(self):
            # pass
            return DataLoader(
                 self.trn_set,
                 batch_size=self.args.batch_size, 
                 shuffle=True,
                 drop_last = True,
                 num_workers=4,
                 collate_fn=collate_fn
                 )

        def val_dataloader(self):
                pass

        def test_dataloader(self):         
                pass

        def predict_dataloader(self):
                pass
      


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, tokenizer,jsonfile):
        self.tokenizer = tokenizer
        self.file_list = json.load(open(jsonfile, "r", encoding="utf-8"))

    def __getitem__(self, index):
        answer_tokens = self.file_list[index]["answer_tokens"]
        conv = conv_llama_3.copy()
        conv.append_message(conv.roles[0], "<speech>\nPlease directly answer the questions in the user's speech.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        speech = whisper.load_audio(self.file_list[index]["wav_path"])
        # default mel
        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)

        return prompt, speech, torch.LongTensor([speech.shape[0]]),answer_tokens
    
    def __len__(self):
        return len(self.file_list)
    

def collate_fn(batch):
    input_ids, speech_tensors, speech_lengths,answer_tokens = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    speech_tensors = torch.stack(speech_tensors, dim=0)
    speech_lengths = torch.stack(speech_lengths, dim=0)
    return input_ids, speech_tensors, speech_lengths,answer_tokens

