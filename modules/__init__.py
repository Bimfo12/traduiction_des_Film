"""
Modules du pipeline de traduction de film.
"""

from .step1_extract_audio import extraire_audio
from .audio_separator import separer_voix_musique, mixer_musique_et_voix
from .step2_diarization import separer_acteurs
from .step3_translation import transcrire_et_traduire
from .step4_voice_cloning import VoiceManager, generer_doublage
from .step5_merge_video import fusionner_video_audio

__all__ = [
    "extraire_audio",
    "separer_voix_musique",
    "mixer_musique_et_voix",
    "separer_acteurs",
    "transcrire_et_traduire",
    "VoiceManager",
    "generer_doublage",
    "fusionner_video_audio",
]
