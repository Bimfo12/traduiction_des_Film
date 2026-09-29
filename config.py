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
WINDOW_SECONDS = 2.0
STEP_SECONDS = 1.0
ENERGY_THRESHOLD = 0.01

# Paramètres de clustering des voix
CLUSTER_DISTANCE_THRESHOLD = 0.35  # Seuil pour regrouper les voix similaires
NUMBER_OF_SPEAKERS = None          # None pour détection automatique, ou entier si connu

# Durée max de la référence vocale par acteur (secondes)
REFERENCE_MAX_SECONDS = 20.0

# Fusion de segments proches d'un même locuteur
MERGE_GAP_SECONDS = 0.35
MIN_SEGMENT_SECONDS = 0.50

# ---------------------------------------------------------------------------
# Étape 3 : Transcription (Whisper) & Traduction (MarianMT)
# ---------------------------------------------------------------------------
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "fr"

# Modèle Whisper ("tiny", "base", "small", "medium", "large-v3")
WHISPER_MODEL = "small"
WHISPER_COMPUTE_TYPE = "float16" if USE_CUDA else "int8"
WHISPER_BEAM_SIZE = 5

# Modèle de traduction Hugging Face MarianMT
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-fr"

# ---------------------------------------------------------------------------
# Étape 4 : Synthèse vocale & Clonage de Voix
# ---------------------------------------------------------------------------
# "xtts" (clonage vocal de l'acteur via XTTS v2) ou "gtts" (Google TTS)
TTS_ENGINE = "xtts"
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_SAMPLE_RATE = 24000

# ---------------------------------------------------------------------------
# Étape 5 : Fusion Vidéo + Audio (FFmpeg)
# ---------------------------------------------------------------------------
AUDIO_BITRATE = "192k"

