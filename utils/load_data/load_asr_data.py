import torchaudio
from torch.utils.data import Dataset, DataLoader
import torchaudio.compliance.kaldi as k
import torchaudio.transforms as T
from transformers import AutoTokenizer
import sys,torch
sys.path.append("./")
from torch.utils.data import Dataset,DataLoader,DistributedSampler
import lightning as L
import json
from torch.nn.utils.rnn import pad_sequence



class data_module(L.LightningDataModule):
        def __init__(self,args):
                super().__init__()
                self.args = args
                self.batch_size = self.args.basic_settings.batch_size

        def setup(self, stage: str):
            # Assign train/val datasets for use in dataloaders
            if stage == "fit":
                # pass
                self.trn_set = CustomDataset(
                    self.args.data_settings.train,
                    self.args.fbank_conf
                    )
                self.eval_set = CustomDataset(
                     self.args.data_settings.eval,
                    self.args.fbank_conf
                    )
   
            # Assign test dataset for use in dataloader(s)
            if stage == "test":
                # pass
                self.test_set = CustomDataset(
                     self.args.data_settings.eval,
                    self.args.fbank_conf
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
                 collate_fn=ASR_collate_fn
                 )

        def val_dataloader(self):
            # pass
            return DataLoader(
                 self.eval_set,
                 batch_size=self.batch_size, 
                 shuffle=False,
                 drop_last = True,
                 num_workers=4,
                 collate_fn=ASR_collate_fn
                 )

        def test_dataloader(self):         
            # pass
            return DataLoader(
                 self.test_set,
                 batch_size=self.batch_size, 
                 shuffle=False,
                 drop_last = False,
                 num_workers=4,
                 collate_fn=ASR_collate_fn
                 )

        def predict_dataloader(self):
                pass
      


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self,jsonfile, fbank_conf):
        #   jsonfile content
        #  {speech_path:speech_path, transcription:transcription}
        self.file_list = json.load(open(jsonfile, "r", encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen2-7B-Instruct", 
            trust_remote_code=True)
        self.fbank_conf = fbank_conf

    def __getitem__(self, index):
        # Load audio
        speech_path = self.file_list[index]["speech_path"]
        fbank = extract_fbank_features(
             speech_path, 
             frame_length=self.fbank_conf.frame_length, 
             frame_shift=self.fbank_conf.frame_shift, 
             sample_rate=self.fbank_conf.sample_rate, 
             n_mels=self.fbank_conf.num_mel_bins
             )
        fbank_len = fbank.shape[0]
        lower_text = self.file_list[index]["transcription"].lower()
        token = self.tokenizer.encode(
             lower_text,
             return_tensors="pt") 
        token_len = token.shape[1]
        # print(self.file_list[index]["transcription"])
        return fbank, fbank_len, token, token_len, lower_text
    
    def __len__(self):
        return len(self.file_list)
    

def extract_fbank_features(
        audio_path, frame_length=25, frame_shift=10, 
        sample_rate=16000, n_mels=80, dither = 0
        ):
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    xs = k.fbank(waveform, 
            dither=dither, 
            frame_length=frame_length, frame_shift=frame_shift, num_mel_bins=n_mels)
    return xs


def ASR_collate_fn(batch):
    fbank_batch = []
    fbank_len_batch = []
    token_batch = []
    token_len_batch = []
    texts = []
    for fbank, fbank_len, token, token_len,text in batch:
        texts.append(text)
        fbank_batch.append(fbank)  # 将FBANK特征转换为tensor
        fbank_len_batch.append(torch.tensor(fbank_len, dtype=torch.long))  # 将fbank_len转换为tensor
        token_batch.append(token.squeeze(0))  # 去掉多余的维度
        token_len_batch.append(torch.tensor(token_len, dtype=torch.long))  # 将token_len转换为tensor
    
    # padding
    fbank_batch_padded = pad_sequence(fbank_batch, batch_first=True, padding_value=0)
    # tokenizer.pad_token_id = 151643
    token_batch_padded = pad_sequence(token_batch, batch_first=True, padding_value=151643)

    fbank_len_batch = torch.stack(fbank_len_batch, dim=0)
    token_len_batch = torch.stack(token_len_batch, dim=0)
    
    return fbank_batch_padded, fbank_len_batch, token_batch_padded, token_len_batch, texts


def post_decode(output, temperature=0.8, top_k=20, top_p=0.8):
        """
        Decoding function, based on the posterior probability output, 
        uses top_k, top_p, and temperature parameters for sampling.

        Parameters:
        - output: torch.Tensor, shaped as (1, 1, D), represents the posterior probability output by the model.
        - top_k: int, indicates selecting the top k tokens with the highest probability for sampling.
                      If 0, no top_k filtering is performed.
        - top_p: float, indicates selecting tokens with cumulative probability not exceeding p for sampling.
                        If 0.0, no top_p filtering is performed.
        - temperature: float, represents the sampling temperature parameter. 
                              The higher the value, the more random the sampling; 
                            the lower the value, the more deterministic the sampling.

        Returns:
        - Selected token index.
        """
        outputs = output
        res = []
        all_res = []
        # print("============="+outputs.shape)
        for i in range(outputs.shape[0]):
            for j in range(outputs.shape[1]):
                output = outputs[i, j, :]
                output = output.squeeze(0).squeeze(0)

                # temperature
                if temperature != 1.0:
                    output = output / temperature

                probs = torch.nn.functional.softmax(output, dim=-1)

                # top_k
                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    top_k_indices.to(dtype=torch.long)
                    probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
                    probs = probs / probs.sum()

                # top_p
                if top_p > 0.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    if sorted_indices_to_remove[0]:
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0
                    probs = probs / probs.sum()

                token_index = torch.multinomial(probs, 1)
                res.append(token_index[0])
            all_res.append(torch.stack(res))
            res = []
        return torch.stack(all_res)



def ctc_greedy_decode(logits, tokenizer):
    """
    对CTC输出进行贪心解码。

    Args:
    logits (torch.Tensor): 形状为 [batch_size, time_steps, vocab_size] 的模型输出。
    tokenizer (tokenizer): 用于将索引转换为 token 的分词器。

    Returns:
    list: 由每个样本的 token 序列组成的列表。
    """
    # 选择最大概率的 token 索引，获取形状为 [batch_size, time_steps] 的最大值索引
    probs, token_ids = torch.max(logits, dim=-1)  # [batch_size, time_steps]

    # 对每个样本进行去重
    decoded_sequences = []
    for i in range(token_ids.size(0)):  # 遍历每个batch
        tokens = token_ids[i].cpu().numpy()  # 取出该batch的token序列
        # 去除重复的token
        tokens = [t for j, t in enumerate(tokens) if j == 0 or t != tokens[j - 1]]
        # 将token索引转为实际的token
        decoded_sequence = tokenizer.decode(tokens).replace("<|endoftext|>"," ")
        decoded_sequences.append(decoded_sequence)
    
    return decoded_sequences, token_ids


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
    conf = toolcfg.yaml2namespace("config/stage_1a.yaml")
    # dm_here = data_module(conf)
    # dm_here.setup("fit")
    # trn_loader = dm_here.train_dataloader()
    # for ele in trn_loader:
    #     print(ele[0].shape)
    #     print(ele[1])
    #     break

    print(type(float(conf.basic_settings.lr)))
    print(float(conf.basic_settings.lr)==0.00001)