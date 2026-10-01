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

    video_path = Path(video_path).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Vidéo introuvable : {video_path}"
        )

    if audio_path is None:
        audio_path = video_path.with_name(
            f"{video_path.stem}_audio.wav"
        )
    else:
        audio_path = Path(audio_path).resolve()

    audio_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    command = [
        "ffmpeg",
        "-y",

        "-i", str(video_path),

        # Première piste audio uniquement
        "-map", "0:a:0",

        # Pas de vidéo, sous-titres ou données
        "-vn",
        "-sn",
        "-dn",

        # WAV PCM 16-bit
        "-c:a", "pcm_s16le",

        # Format de travail principal
        "-ar", str(sample_rate),
        "-ac", str(channels),

        str(audio_path)
    ]

    print("=" * 60)
    print("🎬 ÉTAPE 1 : EXTRACTION DE L'AUDIO DU FILM")
    print("=" * 60)
    print(f"Vidéo source : {video_path}")
    print(f"Audio cible  : {audio_path}")
    print(f"Sample rate  : {sample_rate} Hz")
    print(f"Canaux       : {channels}")
    print("⏳ Extraction en cours avec FFmpeg...")

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

    except subprocess.CalledProcessError as e:
        print("❌ Erreur FFmpeg :")
        print(e.stderr)

        raise RuntimeError(
            f"Échec extraction audio : {e.stderr}"
        ) from e

    except FileNotFoundError as e:
        raise FileNotFoundError(
            "FFmpeg introuvable dans le PATH."
        ) from e

    # ---------------------------------------------------------
    # Vérification du fichier produit
    # ---------------------------------------------------------
    if not audio_path.is_file():
        raise RuntimeError(
            "FFmpeg s'est terminé mais le fichier audio "
            "n'a pas été créé."
        )

    # ---------------------------------------------------------
    # Vérification des durées
    # ---------------------------------------------------------
    try:
        video_probe = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        audio_probe = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        video_duration = float(
            video_probe.stdout.strip()
        )

        audio_duration = float(
            audio_probe.stdout.strip()
        )

        difference = audio_duration - video_duration

        print(
            f"⏱️ Durée vidéo : {video_duration:.3f}s"
        )

        print(
            f"⏱️ Durée audio : {audio_duration:.3f}s"
        )

        print(
            f"📐 Écart      : {difference:+.3f}s"
        )

        if abs(difference) > 0.05:
            print(
                "⚠️ Attention : écart de durée supérieur à 50 ms."
            )

    except Exception as e:
        print(
            f"⚠️ Impossible de vérifier les durées : {e}"
        )

    print(
        f"✅ Audio extrait avec succès : {audio_path}"
    )

    return audio_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python step1_extract_audio.py <chemin_video> [chemin_audio_sortie]")
        sys.exit(1)

    video_input = sys.argv[1]
    audio_output = sys.argv[2] if len(sys.argv) >= 3 else None
    extraire_audio(video_input, audio_output)

