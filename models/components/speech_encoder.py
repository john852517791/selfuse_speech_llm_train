# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/encoder.py

import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
import sys
sys.path.append("./")
from models.components.speech_projector import EncoderProjectorConcat


class WhisperEncoder_adaptor(nn.Module):
    def __init__(self,model_config):
        super().__init__()
        self.encoder = self.load_encoder(model_config)
        self.speech_projector = EncoderProjectorConcat()

    def prepare_speech(
        self, speech, speech_lengths
    ):
        speech_token = self.encoder(speech.permute(0, 2, 1))
        speech_lengths = (speech_lengths + 1) // 2
        encoder_outs_proj = self.speech_projector(speech_token)
        speech_lengths = speech_lengths // self.speech_projector.k
        speech_features = [encoder_outs_proj[i, :speech_lengths[i]] for i in range(len(encoder_outs_proj))]
        return speech_features

    def load_encoder(self,model_config):
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
    

encoder_type = {
    "llama_omni": WhisperEncoder_adaptor,
    # "glm4_encoder": glm4_whisper_encoder,
}

if __name__ == "__main__":
    speech = whisper.load_audio("test/rap01.wav")
    # default mel
    speech = whisper.pad_or_trim(speech)
    speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)

    encoder = encoder_type["llama_omni"]("reference/LLaMA-Omni-main/models/speech_encoder/large-v3.pt").to("cuda")
    op = encoder.prepare_speech(speech.unsqueeze(0).to("cuda"),torch.LongTensor([speech.shape[0]]).unsqueeze(0).to("cuda"))
    print(op)