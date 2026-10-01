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
    original_audio_file: str | Path | None = None,
    music_volume: float = 0.70,
    voice_volume: float = 1.00,
) -> Path:
    """
    Mélange la piste sans voix de Demucs avec la nouvelle voix française.

    La durée finale doit idéalement être celle de l'audio original du film.
    """

    instrumental_file = Path(instrumental_file).resolve()
    voice_file = Path(voice_file).resolve()
    output_file = Path(output_file).resolve()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not instrumental_file.is_file():
        raise FileNotFoundError(
            f"Piste instrumentale introuvable : {instrumental_file}"
        )

    if not voice_file.is_file():
        raise FileNotFoundError(
            f"Piste vocale introuvable : {voice_file}"
        )

    # ---------------------------------------------------------
    # Durée de référence
    # ---------------------------------------------------------
    if original_audio_file is not None:
        original_audio_file = Path(original_audio_file).resolve()

        if not original_audio_file.is_file():
            raise FileNotFoundError(
                f"Audio original introuvable : {original_audio_file}"
            )

        original_duration = get_audio_duration(original_audio_file)

    else:
        # Fallback : prendre la plus longue des deux pistes
        instrumental_duration = get_audio_duration(
            instrumental_file
        )

        voice_duration = get_audio_duration(
            voice_file
        )

        original_duration = 0.0
        duration = max(
            instrumental_duration,
            voice_duration
        )

    duration = max(
        original_duration,
        get_audio_duration(instrumental_file),
        get_audio_duration(voice_file),
    )

    if duration <= 0:
        raise RuntimeError(
            "Impossible de déterminer la durée finale."
        )

    print("\n" + "=" * 60)
    print("🎚️ MIXAGE FINAL : MUSIQUE + VOIX VF")
    print("=" * 60)

    print(f"Musique / ambiance : {instrumental_file}")
    print(f"Voix française     : {voice_file}")
    print(f"Audio original     : {original_audio_file}")
    print(f"Durée finale       : {duration:.2f}s")
    print(f"Volume musique     : {music_volume:.2f}")
    print(f"Volume voix        : {voice_volume:.2f}")

    # ---------------------------------------------------------
    # Filtre
    # ---------------------------------------------------------
    filter_complex = (
        f"[0:a]"
        f"volume={music_volume:.3f},"
        f"aresample=44100,"
        f"asetpts=PTS-STARTPTS"
        f"[music];"

        f"[1:a]"
        f"volume={voice_volume:.3f},"
        f"aresample=44100,"
        f"asetpts=PTS-STARTPTS"
        f"[voice];"

        f"[music][voice]"
        f"amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,"
        f"alimiter=limit=0.90:level=disabled,"
        f"atrim=duration={duration:.6f},"
        f"asetpts=PTS-STARTPTS"
        f"[out]"
    )

    command = [
        "ffmpeg",
        "-y",

        "-i", str(instrumental_file),
        "-i", str(voice_file),

        "-filter_complex",
        filter_complex,

        "-map", "[out]",

        "-ar", "44100",
        "-ac", "2",

        "-c:a", "pcm_s16le",

        "-t", f"{duration:.6f}",

        str(output_file),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            "Erreur FFmpeg pendant le mixage."
        )

    final_duration = get_audio_duration(output_file)

    print(
        f"\n✅ Mixage terminé : {output_file}"
    )
    print(
        f"✅ Durée finale réelle : {final_duration:.2f}s"
    )

    return output_file
