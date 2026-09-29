"""
Étape 4 : Synthèse vocale & Clonage de Voix des Acteurs.
Prend en charge :
- Google Text-to-Speech (gTTS) comme dans main_via_gpt.py (rapide, voix claire, sans crash)
- XTTS-v2 pour le clonage vocal avec patch torchaudio/CUDA
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Permet l'import direct de config quel que soit le dossier d'exécution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment
from gtts import gTTS

import config

# Patch de sécurité pour torchaudio.load afin d'éviter l'erreur libnvrtc.so.13 / torchcodec
_original_torchaudio_load = torchaudio.load

def _safe_torchaudio_load(filepath, *args, **kwargs):
    try:
        data, sr = sf.read(str(filepath), dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)
        return waveform, sr
    except Exception:
        return _original_torchaudio_load(filepath, *args, **kwargs)

torchaudio.load = _safe_torchaudio_load


class VoiceManager:
    """
    Gestionnaire de synthèse vocale pour les acteurs du film.
    Supporte gTTS (comme dans main_via_gpt.py) et XTTS v2 (clonage vocal).
    """

    def __init__(
        self,
        voices: Dict[str, str] | None = None,
        engine: str = config.TTS_ENGINE,
        model_name: str = config.XTTS_MODEL_NAME,
        gpu: bool = config.USE_CUDA
    ):
        self.engine = engine.lower()
        self.voices = voices or {}
        self.xtts = None

        if self.engine == "xtts":
            print("Chargement du modèle de clonage vocal XTTS v2...")
            try:
                from TTS.api import TTS
                self.xtts = TTS(
                    model_name=model_name,
                    progress_bar=False,
                    gpu=gpu
                )
                print("✓ XTTS v2 initialisé.")
            except Exception as e:
                print(f"⚠️ Impossible d'initialiser XTTS ({e}), bascule sur gTTS (Google TTS).")
                self.engine = "gtts"
        else:
            print("✓ Moteur de synthèse vocale : Google Text-to-Speech (gTTS) - main_via_gpt.py")

    def register_voice(self, speaker: str, voice_file: str | Path):
        """Enregistre ou met à jour la référence vocale d'un acteur."""
        self.voices[speaker] = str(Path(voice_file).resolve())

    def speak(
        self,
        speaker: str,
        text: str,
        output_file: str | Path,
        language: str = config.TARGET_LANGUAGE
    ) -> Path:
        """
        Génère un audio français pour l'acteur spécifié.
        """
        output_path = Path(output_file).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not text or not text.strip():
            raise ValueError("Le texte à synthétiser est vide.")

        # Méthode 1 : XTTS v2 (Clonage vocal de l'acteur)
        if self.engine == "xtts" and self.xtts is not None:
            voice_file = self.voices.get(speaker)
            if voice_file and os.path.exists(voice_file):
                try:
                    self.xtts.tts_to_file(
                        text=text,
                        speaker_wav=voice_file,
                        language=language,
                        file_path=str(output_path)
                    )
                    return output_path
                except Exception as e:
                    print(f"  ⚠️ Échec XTTS pour {speaker} ({e}), repli automatique sur gTTS.")

        # Méthode 2 : gTTS (Google TTS comme dans main_via_gpt.py)
        temp_mp3 = output_path.with_suffix(".mp3")
        speech = gTTS(text=text, lang=language, slow=False)
        speech.save(str(temp_mp3))

        # Convertir en WAV 24kHz mono PCM 16-bit
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(temp_mp3),
                "-ar", "24000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(output_path)
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        if temp_mp3.exists() and temp_mp3 != output_path:
            temp_mp3.unlink()

        return output_path


def fit_audio_duration(
    input_path: str | Path,
    output_path: str | Path,
    target_duration_ms: float
) -> bool:
    """
    Ajuste la durée d'un segment audio généré pour qu'il corresponde exactement
    au timing du segment original dans le film (via atempo FFmpeg).
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    probe = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(input_path)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe impossible : {probe.stderr}")

    source_duration_seconds = float(probe.stdout.strip())
    target_duration_seconds = target_duration_ms / 1000.0

    if target_duration_seconds <= 0:
        return False

    tempo = source_duration_seconds / target_duration_seconds
    filters = []

    # atempo accepte entre 0.5 et 2.0 par filtre
    while tempo > 2.0:
        filters.append("atempo=2.0")
        tempo /= 2.0

    while tempo < 0.5:
        filters.append("atempo=0.5")
        tempo /= 0.5

    filters.append(f"atempo={tempo:.6f}")
    filter_string = ",".join(filters)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-filter:a", filter_string,
            "-ar", "24000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(output_path)
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )
    return True


def generer_doublage(
    dialogues: List[Dict],
    speaker_references: Dict[str, str],
    original_audio_path: str | Path,
    output_dir: str | Path,
    final_audio_path: str | Path | None = None,
    engine: str = config.TTS_ENGINE
) -> Path:
    """
    Génère les répliques en français (via gTTS ou XTTS), ajuste leurs durées,
    et les superpose sur la piste audio complète synchronisée.
    """
    output_dir = Path(output_dir).resolve()
    tts_dir = output_dir / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    if final_audio_path is None:
        final_audio_path = output_dir / "film_francais.wav"
    else:
        final_audio_path = Path(final_audio_path).resolve()

    print("=" * 60)
    print("🎙️ ÉTAPE 4 : SYNTHÈSE VOCALE & ASSEMBLAGE DE LA BANDE-SON FR")
    print("=" * 60)
    print(f"Moteur TTS : {engine.upper()}")
    print(f"Nombre de répliques à doubler : {len(dialogues)}")
    print(f"Acteurs référencés : {len(speaker_references)}")

    # Mesure de la durée originale
    original_audio = AudioSegment.from_file(str(original_audio_path))
    total_duration_ms = len(original_audio)
    print(f"Durée totale du film : {total_duration_ms / 1000:.2f} secondes")

    # Initialisation de VoiceManager
    voice_manager = VoiceManager(voices=speaker_references, engine=engine, gpu=config.USE_CUDA)

    dubbed_segments = []
    total = len(dialogues)

    for i, dialog in enumerate(dialogues, start=1):
        idx = dialog["index"]
        speaker = dialog["speaker"]
        french_text = dialog["french_text"]
        start_sec = dialog["start"]
        end_sec = dialog["end"]
        duration_sec = dialog["duration"]

        print(f"\n[{i}/{total}] Voix pour {speaker} ({start_sec:.2f}s -> {end_sec:.2f}s, {duration_sec:.2f}s)")
        print(f"  Texte FR : \"{french_text}\"")

        raw_tts_file = tts_dir / f"tts_{idx:05d}_{speaker}_raw.wav"
        fitted_tts_file = tts_dir / f"tts_{idx:05d}_{speaker}.wav"

        try:
            # 1. Synthèse vocale
            voice_manager.speak(
                speaker=speaker,
                text=french_text,
                output_file=raw_tts_file,
                language=config.TARGET_LANGUAGE
            )

            # 2. Ajustement de la durée pour caler au timing exact
            target_duration_ms = duration_sec * 1000.0
            fit_audio_duration(raw_tts_file, fitted_tts_file, target_duration_ms)

            dubbed_segments.append({
                "index": idx,
                "speaker": speaker,
                "start_ms": int(start_sec * 1000),
                "duration_ms": int(target_duration_ms),
                "audio_path": str(fitted_tts_file)
            })
            print(f"  ✓ Réplique française générée et calée.")

        except Exception as e:
            print(f"  ❌ Erreur génération réplique {idx} ({speaker}): {e}")

    # 3. Réassemblage de la piste globale synchronisée
    print(f"\nAssemblage final de la piste audio française ({len(dubbed_segments)} répliques assemblées)...")
    final_audio = AudioSegment.silent(duration=total_duration_ms, frame_rate=24000)

    for seg in dubbed_segments:
        seg_audio_file = seg["audio_path"]
        if not os.path.exists(seg_audio_file):
            continue

        speech = AudioSegment.from_file(seg_audio_file)
        start_ms = seg["start_ms"]
        duration_ms = seg["duration_ms"]

        if len(speech) > duration_ms:
            speech = speech[:duration_ms]

        final_audio = final_audio.overlay(speech, position=start_ms)

    final_audio.export(str(final_audio_path), format="wav")
    print(f"✅ Bande audio doublée en français prête : {final_audio_path}")
    return final_audio_path


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python step4_voice_cloning.py <dialogues_traduits.json> <speakers_info.json> <audio_original.wav> [audio_sortie.wav]")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f_d:
        d_input = json.load(f_d)

    with open(sys.argv[2], "r", encoding="utf-8") as f_s:
        s_info = json.load(f_s)
        s_refs = s_info.get("references", {})

    orig_audio = sys.argv[3]
    out_audio = sys.argv[4] if len(sys.argv) >= 5 else None

    generer_doublage(d_input, s_refs, orig_audio, Path(sys.argv[1]).parent, out_audio)
