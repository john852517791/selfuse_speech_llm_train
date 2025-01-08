import torchaudio
from torch.utils.data import Dataset, DataLoader
import torchaudio.compliance.kaldi as k
import torchaudio.transforms as T
from transformers import AutoTokenizer,AutoModelForCausalLM
import sys,torch
sys.path.append("./")
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from models.decoder.ticodec.vqvae_tester import VqvaeTester
import lightning as L
import json
from torch.nn.utils.rnn import pad_sequence



class data_module(L.LightningDataModule):
        def __init__(self,args):
                super().__init__()
                self.args = args
                self.batch_size = self.args.basic_settings.batch_size
                self.llm_path = self.args.basic_settings.llm_path

        def setup(self, stage: str):
            # Assign train/val datasets for use in dataloaders
            if stage == "fit":
                # pass
                self.trn_set = CustomDataset(
                    self.args.data_settings.train,
                    self.llm_path
                    )
                self.eval_set = CustomDataset(
                    self.args.data_settings.eval,
                    self.llm_path
                    )
   
            # Assign test dataset for use in dataloader(s)
            if stage == "test":
                # pass
                self.test_set = CustomDataset(
                    self.args.data_settings.eval,
                    self.llm_path
                    )

            if stage == "predict":
                pass
                    
                    
                

        def train_dataloader(self):
            # pass
            return DataLoader(
                 self.trn_set,
                 batch_size=self.batch_size, 
                 shuffle=True,
                 drop_last = True,
                 num_workers=4,
                 collate_fn=tts_collate_fn
                 )

        def val_dataloader(self):
            # pass
            return DataLoader(
                 self.eval_set,
                 batch_size=self.batch_size, 
                 shuffle=False,
                 drop_last = True,
                 num_workers=4,
                 collate_fn=tts_collate_fn
                 )

        def test_dataloader(self):         
            # pass
            return DataLoader(
                 self.test_set,
                 batch_size=self.batch_size, 
                 shuffle=False,
                 drop_last = False,
                 num_workers=4,
                 collate_fn=tts_collate_fn
                 )

        def predict_dataloader(self):
                pass
      


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self,jsonfile, llm_path):
        #   jsonfile content
        #  {speech_path:speech_path, transcription:transcription}
        # speech -> codec tokens
        # text -> llm tokens -> llm embedding
        self.file_list = json.load(open(jsonfile, "r", encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path, 
            trust_remote_code=True)
        self.LLM_embedding_layer = AutoModelForCausalLM.from_pretrained(
            llm_path, 
            torch_dtype=torch.float32,
            trust_remote_code=True).model.embed_tokens
        model_path = "Freeze-Omni/checkpoints"
        self.codec_model = VqvaeTester(config_path=model_path + "/codec/model.json", 
                                                model_path=model_path + "/codec/final.pt",
                                                sample_rate=24000)
        # self.codec_model = self.codec_model.cuda()
        self.codec_model.vqvae.generator.remove_weight_norm()
        self.codec_model.vqvae.encoder.remove_weight_norm()
        self.codec_model.eval()


    def __getitem__(self, index):
        # speech -> codec tokens
        # text -> llm tokens -> llm embedding
        lower_text = self.file_list[index]["transcription"].lower()
        token = self.LLM_embedding_layer(self.tokenizer.encode(
             lower_text,
             return_tensors="pt")).reshape(-1, 896).detach()
        token_len = token.shape[0]

        speech_path = self.file_list[index]["speech_path"]
        fid, vq_codes, global_token = self.codec_model.vq(speech_path)
        vq_codes_len = len(vq_codes[0])

        # print(self.file_list[index]["transcription"])
        return vq_codes, vq_codes_len, token, token_len, lower_text, global_token
    
    def __len__(self):
        return len(self.file_list)



def tts_collate_fn(batch):
    all_vq_codes = []
    all_vq_codes_len = []
    all_token = []
    all_token_len = []
    all_lower_text = []
    all_global_token = []
    for vq_codes, vq_codes_len, token, token_len, lower_text, global_token in batch:
        all_vq_codes.append(vq_codes.squeeze(0).squeeze(-1))
        all_vq_codes_len.append(torch.tensor(vq_codes_len, dtype=torch.long))
        all_token.append(token.squeeze(0))
        all_token_len.append(torch.tensor(token_len, dtype=torch.long))
        all_lower_text.append(lower_text)
        all_global_token.append(global_token.squeeze(0))
    
    # padding
    all_vq_codes_padded = pad_sequence(all_vq_codes, batch_first=True, padding_value=0)
    all_token_padded = pad_sequence(all_token, batch_first=True, padding_value=0)
    all_global_token_padded = pad_sequence(all_global_token, batch_first=True, padding_value=0)
    # tokenizer.pad_token_id = 151643

    all_vq_codes_len_batch = torch.stack(all_vq_codes_len, dim=0)
    all_token_len_batch = torch.stack(all_token_len, dim=0)
    
    return all_vq_codes_padded, all_vq_codes_len_batch, all_token_padded, all_token_len_batch, all_global_token_padded, all_lower_text


if __name__ == "__main__":
    # asrdataset = CustomDataset("utils/load_data/asrdata.json")
    # asrDL = DataLoader(
    #              asrdataset,
    #              batch_size=2, 
    #              shuffle=True,
    #              drop_last = True,
    #              num_workers=4,
    #              collate_fn=ASR_collate_fn
    #              )
    # for ele in asrDL:
    #      print(ele)
    #      break
    import utils.tools.config as toolcfg
    conf = toolcfg.yaml2namespace("config/stage_2a.yaml")
    dm_here = data_module(conf)
    dm_here.setup("fit")
    trn_loader = dm_here.train_dataloader()
    for ele in trn_loader:
        print(ele[0].shape)
        print(ele[1])
        break

    # print(type(float(conf.basic_settings.lr)))
    # print(float(conf.basic_settings.lr)==0.00001)