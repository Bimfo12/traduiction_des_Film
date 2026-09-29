"""
Étape 1 : Séparation de l'audio de la vidéo.
Basé sur convert_video_to_audio/convert_video_to_audio.py
"""

import os
import sys
import subprocess
from pathlib import Path


def extraire_audio(
    video_path: str | Path,
    audio_path: str | Path | None = None,
    sample_rate: int = 44100,
    channels: int = 2
) -> Path:
    """
    Extrait l'audio d'une vidéo et le sauvegarde en format WAV non compressé (PCM 16-bit).

    Args:
        video_path: Chemin vers le fichier vidéo source.
        audio_path: Chemin du fichier audio de sortie (optionnel).
        sample_rate: Fréquence d'échantillonnage (défaut: 44100 Hz).
        channels: Nombre de canaux (1 = mono, 2 = stéréo).

    Returns:
        Path vers le fichier audio extrait.
    """
    video_path = Path(video_path).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {video_path}")

    if audio_path is None:
        audio_path = video_path.with_name(f"{video_path.stem}_audio.wav")
    else:
        audio_path = Path(audio_path).resolve()

    audio_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",                       # Supprimer le flux vidéo
        "-acodec", "pcm_s16le",      # PCM 16-bit non compressé
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-y",                        # Écraser si existant
        str(audio_path)
    ]

    print("=" * 60)
    print("🎬 ÉTAPE 1 : EXTRACTION DE L'AUDIO DU FILM")
    print("=" * 60)
    print(f"Vidéo source : {video_path}")
    print(f"Audio cible  : {audio_path}")
    print("⏳ Extraction en cours avec FFmpeg...")

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"✅ Audio extrait avec succès : {audio_path}")
        return audio_path

    except subprocess.CalledProcessError as e:
        print("❌ Erreur lors de l'extraction FFmpeg :")
        print(e.stderr)
        raise RuntimeError(f"Échec de l'extraction audio FFmpeg: {e.stderr}") from e

    except FileNotFoundError as e:
        print("❌ FFmpeg n'est pas installé ou n'est pas accessible dans le PATH.")
        raise FileNotFoundError("FFmpeg introuvable dans le PATH.") from e


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python step1_extract_audio.py <chemin_video> [chemin_audio_sortie]")
        sys.exit(1)

    video_input = sys.argv[1]
    audio_output = sys.argv[2] if len(sys.argv) >= 3 else None
    extraire_audio(video_input, audio_output)

