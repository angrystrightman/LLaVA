import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


from .whisper_encoder import WhisperEncoder  # Import the new WhisperEncoder class

def build_audio_tower(audio_tower_cfg, **kwargs):
     # Determine the type of encoder to use
    audio_tower = getattr(audio_tower_cfg, 'mm_audio_tower', getattr(audio_tower_cfg, 'audio_tower', None))
    is_absolute_path_exists = os.path.exists(audio_tower)
    
    use_whisper = "whisper" in audio_tower.lower()  # Check if the config indicates the use of Whisper
    #更多encoder待支持
    # Handle the case for Whisper encoder
    if is_absolute_path_exists: #or audio_tower.startswith("openai"):
        if use_whisper:
            return WhisperEncoder(audio_tower, args=audio_tower_cfg, **kwargs)
        
    raise ValueError(f'Unknown audio tower: {audio_tower}')
    

    
