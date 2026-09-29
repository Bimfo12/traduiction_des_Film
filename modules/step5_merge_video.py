"""
Étape 5 : Fusion de la vidéo originale avec la nouvelle piste audio française.
Basé sur fusionner_audio_video/fusionner_audio_video.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Permet l'import direct de config quel que soit le dossier d'exécution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def fusionner_video_audio(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path | None = None,
    bitrate: str = config.AUDIO_BITRATE
) -> Path:
    """
    Fusionne le flux vidéo original (sans réencodage) avec la nouvelle piste audio doublée en AAC.

    Args:
        video_path: Chemin du fichier vidéo original.
        audio_path: Chemin de la nouvelle piste audio française.
        output_path: Chemin de la vidéo finale de sortie.
        bitrate: Débit binaire audio (ex: '192k').

    Returns:
        Path vers la vidéo finale générée.
    """
    video_path = Path(video_path).resolve()
    audio_path = Path(audio_path).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {video_path}")

    if not audio_path.is_file():
        raise FileNotFoundError(f"Piste audio introuvable : {audio_path}")

    if output_path is None:
        output_path = video_path.with_name(f"{video_path.stem}_VF{video_path.suffix}")
    else:
        output_path = Path(output_path).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",       # Flux vidéo du fichier source
        "-map", "1:a:0",       # Flux audio doublé
        "-c:v", "copy",        # Copie directe sans perte de qualité vidéo
        "-c:a", "aac",         # Encodage audio haute fidélité
        "-b:a", bitrate,
        "-shortest",           # Aligne sur la durée
        "-y",
        str(output_path)
    ]

    print("=" * 60)
    print("🎞️ ÉTAPE 5 : FUSION VIDÉO + NOUVELLE PISTE AUDIO DOUBLÉE")
    print("=" * 60)
    print(f"Vidéo originale  : {video_path}")
    print(f"Bande-son FR     : {audio_path}")
    print(f"Vidéo finale VF  : {output_path}")
    print("⏳ Fusion en cours avec FFmpeg...")

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

        print()
        print(f"🎉 SUCCÈS ! Film entièrement traduit et doublé disponible ici :")
        print(f"👉 {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        print("❌ Erreur FFmpeg lors de la fusion :")
        print(e.stderr)
        raise RuntimeError(f"Échec de la fusion FFmpeg : {e.stderr}") from e

    except FileNotFoundError as e:
        print("❌ FFmpeg introuvable dans le PATH.")
        raise FileNotFoundError("FFmpeg introuvable dans le PATH.") from e


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python step5_merge_video.py <video.mp4> <audio.wav> [film_vf.mp4]")
        sys.exit(1)

    v_in = sys.argv[1]
    a_in = sys.argv[2]
    v_out = sys.argv[3] if len(sys.argv) >= 4 else None
    fusionner_video_audio(v_in, a_in, v_out)

