"""
Séparation de la voix et de la musique de fond avec Demucs,
et mixage final de la musique de fond avec la nouvelle voix française.
Basé exactement sur la logique de main_via_gpt.py.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple

# Permet l'import direct de config quel que soit le dossier d'exécution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def get_audio_duration(audio_file: str | Path) -> float:
    """Retourne la durée d'un fichier audio en secondes."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_file),
        ],
        capture_output=True,
        text=True,
        check=True
    )
    try:
        return float(probe.stdout.strip())
    except (ValueError, TypeError):
        return 0.0


def separer_voix_musique(
    audio_path: str | Path,
    output_dir: str | Path,
    model_name: str = "htdemucs",
    device: str = config.DEVICE
) -> Tuple[Path, Path]:
    """
    Sépare l'audio en deux pistes :
    - vocals.wav : la voix parlée originale (sans musique)
    - no_vocals.wav : la musique de fond et effets sonores (sans voix)

    Args:
        audio_path: Fichier audio original complet du film.
        output_dir: Dossier où stocker les pistes séparées.

    Returns:
        Tuple (vocals_path, instrumental_path)
    """
    audio_path = Path(audio_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")

    # Vérification des fichiers déjà extraits pour éviter de recalculer inutilement
    target_vocals = output_dir / "vocals.wav"
    target_instrumental = output_dir / "no_vocals.wav"

    if target_vocals.exists() and target_instrumental.exists():
        print("✓ Pistes voix et musique de fond déjà séparées :")
        print(f"  • Voix     : {target_vocals}")
        print(f"  • Musique  : {target_instrumental}")
        return target_vocals, target_instrumental

    print("\n" + "=" * 60)
    print("🎵 SÉPARATION VOIX / MUSIQUE DE FOND (DEMUCS)")
    print("=" * 60)
    print(f"Fichier audio : {audio_path}")
    print(f"Modèle Demucs : {model_name}")
    print(f"Périphérique  : {device}")
    print("⏳ Séparation en cours avec Demucs...")

    # Utilise le python de l'environnement virtuel pour lancer demucs
    venv_python = config.BASE_DIR / "venv" / "bin" / "python"
    python_cmd = str(venv_python) if venv_python.exists() else sys.executable

    demucs_cmd = [
        python_cmd,
        "-m", "demucs.separate",
        "--two-stems=vocals",
        "-n", model_name,
        "-d", device,
        "--shifts", "1",
        "--overlap", "0.15",
        "-o", str(output_dir),
        str(audio_path)
    ]

    subprocess.run(demucs_cmd, check=True)

    base_name = audio_path.stem
    # Demucs écrit par défaut dans output_dir / model_name / base_name /
    demucs_out_dir = output_dir / model_name / base_name
    src_vocals = demucs_out_dir / "vocals.wav"
    src_instrumental = demucs_out_dir / "no_vocals.wav"

    if not src_vocals.exists() or not src_instrumental.exists():
        raise FileNotFoundError(f"Demucs n'a pas produit les fichiers attendus dans {demucs_out_dir}")

    # Déplacer ou copier vers target_vocals et target_instrumental
    shutil.copy2(src_vocals, target_vocals)
    shutil.copy2(src_instrumental, target_instrumental)

    print(f"\n✅ Séparation terminée avec succès !")
    print(f"  • Voix originale   : {target_vocals}")
    print(f"  • Musique de fond  : {target_instrumental}")
    return target_vocals, target_instrumental


def mixer_musique_et_voix(
    instrumental_file: str | Path,
    voice_file: str | Path,
    output_file: str | Path,
    music_volume: float = 1.0,
    voice_volume: float = 1.0
) -> Path:
    """
    Mélange la musique de fond originale (sans voix anglaise)
    avec la nouvelle piste vocale française (VF).
    Basé sur mix_audio() de main_via_gpt.py.
    """
    instrumental_file = Path(instrumental_file).resolve()
    voice_file = Path(voice_file).resolve()
    output_file = Path(output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not instrumental_file.exists():
        raise FileNotFoundError(f"Piste musicale introuvable : {instrumental_file}")

    if not voice_file.exists():
        raise FileNotFoundError(f"Piste vocale introuvable : {voice_file}")

    duration = get_audio_duration(instrumental_file)
    if duration <= 0:
        duration = get_audio_duration(voice_file)

    print("\n" + "=" * 60)
    print("🎚️ MIXAGE FINAL : MUSIQUE DE FOND + NOUVELLES VOIX VF")
    print("=" * 60)
    print(f"Musique de fond : {instrumental_file}")
    print(f"Voix françaises : {voice_file}")
    print(f"Audio final     : {output_file}")
    print(f"Durée totale    : {duration:.2f}s")

    filter_complex = (
        f"[0:a]volume={music_volume:.2f}[music];"
        f"[1:a]volume={voice_volume:.2f}[voice];"
        f"[music][voice]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[a];"
        f"[a]atrim=duration={duration:.6f},asetpts=PTS-STARTPTS[out]"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i", str(instrumental_file),
        "-i", str(voice_file),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:a", "pcm_s16le",
        "-ar", "44100",
        "-t", str(duration),
        str(output_file)
    ]

    subprocess.run(command, check=True)
    print(f"✅ Bande-son complète mixée (musique + voix VF) : {output_file}")
    return output_file

