"""
Configuration centrale pour la chaîne de traduction et doublage de film.
"""

from pathlib import Path
import torch

# Chemins de base du projet
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WORK_DIR = BASE_DIR / "workspace"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"

# Détection automatique de l'accélérateur GPU
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

# ---------------------------------------------------------------------------
# Étape 1 : Extraction Audio
# ---------------------------------------------------------------------------
AUDIO_EXTRACT_SAMPLE_RATE = 44100
AUDIO_EXTRACT_CHANNELS = 2  # stéréo pour préserver la qualité originale

# ---------------------------------------------------------------------------
# Étape 2 : Diarisation des locuteurs (SpeechBrain)
# ---------------------------------------------------------------------------
SPEECHBRAIN_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SPEECHBRAIN_SAMPLE_RATE = 16000

# Paramètres de détection de voix
WINDOW_SECONDS = 2.00
STEP_SECONDS = 1.00
ENERGY_THRESHOLD = 0.012

# Paramètres de clustering des voix
CLUSTER_DISTANCE_THRESHOLD = 0.42  # Équilibre entre séparation et fusion des locuteurs
NUMBER_OF_SPEAKERS = None          # None pour détection automatique, ou entier si connu

# Durée max de la référence vocale par acteur (secondes)
REFERENCE_MAX_SECONDS = 20.0

# Fusion de segments proches d'un même locuteur
MERGE_GAP_SECONDS = 0.45
MIN_SEGMENT_SECONDS = 0.70

# ---------------------------------------------------------------------------
# Étape 3 : Transcription (Whisper) & Traduction (Malgache / Manuelle)
# ---------------------------------------------------------------------------
# Langue source de la vidéo : "auto" (détecte automatiquement le français ou l'anglais), ou "fr", "en"
SOURCE_LANGUAGE = "auto"

# Langue cible : Malgache ("mg")
TARGET_LANGUAGE = "mg"

# Mode de traduction manuelle : si True, génère le fichier texte avec les répliques
# malgaches laissées vides afin que l'utilisateur puisse les compléter manuellement.
MANUAL_TRANSLATION = True
DIALOGUES_MALGACHE_TXT = "dialogues_malgache.txt"

# Modèle Whisper ("tiny", "base", "small", "medium", "large-v3")
WHISPER_MODEL = "small"
WHISPER_COMPUTE_TYPE = "float16" if USE_CUDA else "int8"
WHISPER_BEAM_SIZE = 5

# Modèle de traduction automatique (optionnel, utilisé uniquement si MANUAL_TRANSLATION est False)
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-fr"

# ---------------------------------------------------------------------------
# Étape 4 : Synthèse vocale & Clonage de Voix
# ---------------------------------------------------------------------------
# "xtts" (clonage vocal de l'acteur via XTTS v2) ou "gtts" (Google TTS, supporte 'mg')
TTS_ENGINE = "xtts"
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
# Pour XTTS, 'mg' n'étant pas une langue native du modèle, on utilise 'fr' ou 'it'
# pour la prononciation phonétique latine fluide de l'alphabet malgache
XTTS_LANGUAGE = "fr"
TTS_SAMPLE_RATE = 24000
TTS_MIN_SPEED = 0.95
TTS_MAX_SPEED = 1.12
REUSE_TTS_CACHE = False

# ---------------------------------------------------------------------------
# Étape 5 : Fusion Vidéo + Audio (FFmpeg)
# ---------------------------------------------------------------------------
AUDIO_BITRATE = "192k"

# ---------------------------------------------------------------------------
# Robustesse : Diarisation & Références vocales
# ---------------------------------------------------------------------------
SEGMENT_GUARD_SECONDS = 0.05
REFERENCE_MIN_SECONDS = 3.0
REFERENCE_TARGET_SECONDS = 7.0
REFERENCE_SILENCE_SECONDS = 0.15

# ---------------------------------------------------------------------------
# Robustesse : Doublage & Assemblage
# ---------------------------------------------------------------------------
DUBBING_GUARD_SECONDS = 0.08
DUBBING_GUARD_MS = int(DUBBING_GUARD_SECONDS * 1000)

# ---------------------------------------------------------------------------
# Robustesse : TTS
# ---------------------------------------------------------------------------
TTS_CHARS_PER_SEC = 14.0

# ---------------------------------------------------------------------------
# Robustesse : Séparation audio
# ---------------------------------------------------------------------------
DEMUCS_MODEL = "htdemucs_ft"
USE_DUCKING = True

# ---------------------------------------------------------------------------
# Diarisation : fusion des clusters de locuteurs trop proches
# ---------------------------------------------------------------------------
# Fusionne après clustering les clusters dont la distance cosine est inférieure
# à ce seuil. 0.25 est un bon compromis (0.0 = désactivé).
SPEAKER_MERGE_DISTANCE_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# TTS : température XTTS pour stabiliser le timbre vocal par acteur
# ---------------------------------------------------------------------------
# 0.30 donne un timbre cohérent entre les répliques. 0.0 = désactivé.
XTTS_TEMPERATURE = 0.30

# ---------------------------------------------------------------------------
# Étape 5 : mode de synchronisation vidéo/audio
# ---------------------------------------------------------------------------
# "lock_video" (défaut) : la vidéo garde sa durée originale, l'audio est padded/trimé.
# "match_audio" : ajuste la vitesse vidéo pour correspondre à la durée de l'audio VF.
VIDEO_SYNC_MODE = "lock_video"
