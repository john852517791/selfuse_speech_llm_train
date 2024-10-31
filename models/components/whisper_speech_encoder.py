# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/encoder.py

import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper


class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):

        def replace_layer_norm(module):
            from whisper.model import LayerNorm
            for name, child in module.named_children():
                if isinstance(child, LayerNorm):
                    old_params = child.state_dict()
                    new_layer_norm = nn.LayerNorm(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
                    new_layer_norm.load_state_dict(old_params)
                    setattr(module, name, new_layer_norm)
                else:
                    replace_layer_norm(child)

        encoder = whisper.load_model(name=model_config).encoder
        replace_layer_norm(encoder)
        return encoder
    
if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("/data8/wangzhiyong/project/LLM/llama_omni/datasets/question_wav/wav")['train']
    import numpy as np
    sample = ds[2]["audio"]
    sample = np.array(sample["array"], dtype=np.float32)
    encoder = WhisperWrappedEncoder.load("/data8/wangzhiyong/project/LLM/llama_omni/reference/LLaMA-Omni-main/models/speech_encoder/large-v3.pt")
    speech = whisper.pad_or_trim(sample)
    print(sample.shape)
    # (31232,)
    speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
    print(speech.shape)
    # torch.Size([3000, 128])
    op = encoder(speech)