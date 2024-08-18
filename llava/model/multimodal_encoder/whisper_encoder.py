import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperProcessor, WhisperConfig

class WhisperEncoder(nn.Module):
    def __init__(self, encoder_name, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.encoder_name = encoder_name
        self.select_layer = args.audio_select_layer  # Similar to mm_vision_select_layer for audio
        self.select_feature = getattr(args, 'audio_select_feature', 'embedding')  # 'embedding' as a placeholder

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_audio_encoder', False):
            self.load_model()
        else:
            self.cfg_only = WhisperConfig.from_pretrained(self.encoder_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.encoder_name))
            return

        self.audio_processor = WhisperProcessor.from_pretrained(self.encoder_name)
        self.encoder = WhisperModel.from_pretrained(self.encoder_name, device_map=device_map)
        self.encoder.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, audio_forward_outs):
        audio_features = audio_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'embedding':
            # Select the embedding features, adapt this part based on how Whisper outputs
            audio_features = audio_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return audio_features

    @torch.no_grad()
    def forward(self, audio):
        if type(audio) is list:
            audio_features = []
            for waveform in audio:
                audio_forward_out = self.encoder(waveform.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                audio_feature = self.feature_select(audio_forward_out).to(waveform.dtype)
                audio_features.append(audio_feature)
        else:
            audio_forward_outs = self.encoder(audio.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            audio_features = self.feature_select(audio_forward_outs).to(audio.dtype)

        return audio_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.encoder.dtype

    @property
    def device(self):
        return self.encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def sample_rate(self):
        return self.config.sampling_rate
