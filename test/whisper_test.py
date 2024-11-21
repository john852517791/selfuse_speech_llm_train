import whisper
import torch.nn as nn
import torch
class WhisperWrappedEncoder:
    @classmethod
    def load(cls):
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
        import whisper
        encoder = whisper.load_model(name="large-v3", device='cpu').encoder
        replace_layer_norm(encoder)
        return encoder

class EncoderProjectorConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = 5
        self.encoder_dim = 1280
        self.llm_dim = 4096
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, 4096)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


speech_projector = EncoderProjectorConcat()
whisper_encoder = WhisperWrappedEncoder.load()
# model = whisper.load_model("/root/wangzhiyong/project/llama_omni_train/models/speech_encoder/large-v3.pt", download_root="models/speech_encoder/")
speech = whisper.load_audio("/root/wangzhiyong/project/llama_omni_train/reference/LLaMA-Omni-main/omni_speech/infer/examples/question_wav/helpful_base_1.wav")
speech = whisper.pad_or_trim(speech)
speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
speech_lengths = torch.LongTensor([speech.shape[0]])
# print(whisper_encoder(mel,length))

encoder_outs = whisper_encoder(speech.unsqueeze(0).permute(0, 2, 1))
speech_lengths = (speech_lengths + 1) // 2
encoder_outs_proj = speech_projector(encoder_outs)
speech_lengths = speech_lengths // speech_projector.k
speech_features = [encoder_outs[i, :speech_lengths[i]] for i in range(len(encoder_outs))]

print(speech_features)


