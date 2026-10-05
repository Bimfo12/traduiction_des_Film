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
from pydub.silence import detect_nonsilent

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
    Supporte XTTS v2 (clonage vocal de l'acteur) et gTTS (Google TTS).
    """

    def __init__(
        self,
        voices: Dict[str, str] | None = None,
        engine: str = config.TTS_ENGINE,
        model_name: str = config.XTTS_MODEL_NAME,
        gpu: bool = config.USE_CUDA,
        segments_dir: str | Path | None = None
    ):
        self.engine = engine.lower()
        self.voices = voices or {}
        self.segments_dir = Path(segments_dir).resolve() if segments_dir else None
        self.cloneur = None
        self.xtts = None

        if self.engine == "xtts":
            print("Chargement du moteur de clonage vocal XTTS v2...")
            try:
                from clonnage_vocal import CloneurVocalActeur
                seg_folder = self.segments_dir or (Path(config.DEFAULT_WORK_DIR) / "video" / "segments")
                self.cloneur = CloneurVocalActeur(
                    segments_dir=seg_folder,
                    model_name=model_name,
                    gpu=gpu
                )
                self.xtts = self.cloneur.tts
                print("✓ Modèle XTTS v2 initialisé pour le clonage de chaque acteur.")
            except Exception as e:
                print(f"⚠️ Échec chargement CloneurVocalActeur ({e}), tentative TTS directe...")
                try:
                    from TTS.api import TTS
                    self.xtts = TTS(
                        model_name=model_name,
                        progress_bar=False,
                        gpu=gpu and torch.cuda.is_available()
                    )
                    print("✓ XTTS v2 direct initialisé.")
                except Exception as e2:
                    raise RuntimeError(
                        f"Impossible d'initialiser XTTS pour cloner les voix : {e2}"
                    ) from e2
        else:
            print("✓ Moteur de synthèse vocale : Google Text-to-Speech (gTTS)")

    def register_voice(self, speaker: str, voice_file: str | Path):
        """Enregistre ou met à jour la référence vocale d'un acteur."""
        self.voices[speaker] = str(Path(voice_file).resolve())

    def speak(
        self,
        speaker: str,
        text: str,
        output_file: str | Path,
        language: str = config.TARGET_LANGUAGE,
        reference_wav: str | Path | None = None
    ) -> Path:
        """
        Génère un audio français en clonant fidèlement la voix de l'acteur.
        Utilise en priorité le segment audio original de l'acteur (speaker_wav).
        """
        output_path = Path(output_file).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not text or not text.strip():
            raise ValueError("Le texte à synthétiser est vide.")

        # Méthode 1 : Clonage vocal de l'acteur via XTTS v2
        if self.engine == "xtts":
            # Priorité absolue : référence consolidée de l'acteur
            ref_source = self.voices.get(speaker)

            # Fallback uniquement si la référence consolidée est absente
            if not ref_source and reference_wav:
                ref_source = reference_wav

            if not ref_source or not os.path.isfile(str(ref_source)):
                raise RuntimeError(
                    f"Aucune référence vocale valide trouvée pour l'acteur {speaker}."
                )

            ref_source = str(Path(ref_source).resolve())

            # Si le cloneur personnalisé est disponible
            if self.cloneur is not None:
                try:
                    self.cloneur.cloner_voix(
                        text=text,
                        reference_segment=ref_source,
                        output_file=output_path,
                        language=language,
                    )
                    return output_path
                except Exception as e:
                    print(f"  ⚠️ CloneurVocalActeur en échec pour {speaker} ({e}), essai XTTS direct...")

            # Repli XTTS direct
            if self.xtts is not None:
                try:
                    self.xtts.tts_to_file(
                        text=text,
                        speaker_wav=ref_source,
                        language=language,
                        file_path=str(output_path)
                    )
                    return output_path
                except Exception as e:
                    raise RuntimeError(
                        f"Échec du clonage XTTS pour {speaker} : {e}"
                    ) from e

            raise RuntimeError(
                f"Aucun extrait de référence valide pour cloner la voix de {speaker}."
            )

        # Méthode 2 : gTTS (Google TTS) en repli
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



def trim_edge_silence(
    audio: AudioSegment,
    silence_thresh_offset: int = 18,
    min_silence_len: int = 100,
    keep_ms: int = 30
) -> AudioSegment:
    """
    Supprime uniquement les silences au début et à la fin.
    Les pauses entre les mots/phrases sont conservées.
    """

    if len(audio) == 0:
        return audio

    # Évite un seuil trop bas sur les fichiers très faibles
    silence_thresh = max(audio.dBFS - silence_thresh_offset, -50)

    ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )

    # Aucun contenu détecté
    if not ranges:
        return audio

    start = max(0, ranges[0][0] - keep_ms)
    end = min(len(audio), ranges[-1][1] + keep_ms)

    return audio[start:end]


def fit_audio_duration(
    input_path: str | Path,
    output_path: str | Path,
    target_duration_ms: float,
    min_speed: float = config.TTS_MIN_SPEED,
    max_speed: float = config.TTS_MAX_SPEED,
) -> tuple[bool, int]:
    """
    Ajuste légèrement le débit d'une voix TTS.

    - 0.90 = maximum ~10 % plus lent
    - 1.15 = maximum ~15 % plus rapide

    Si l'écart est trop important, on privilégie la voix naturelle
    plutôt qu'une accélération/ralentissement extrême.

    Retourne:
        (success, durée_finale_ms)
    """

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Audio introuvable : {input_path}")

    target_duration_ms = float(target_duration_ms)

    if target_duration_ms <= 0:
        return False, 0

    # ---------------------------------------------------------
    # Charger le TTS
    # ---------------------------------------------------------
    audio = AudioSegment.from_file(input_path)

    if len(audio) <= 0:
        raise ValueError("Audio TTS vide")

    # ---------------------------------------------------------
    # Supprimer seulement les silences aux extrémités
    # ---------------------------------------------------------
    audio = trim_edge_silence(
        audio,
        silence_thresh_offset=18,
        min_silence_len=100,
        keep_ms=40,
    )

    source_duration_ms = len(audio)

    if source_duration_ms <= 0:
        raise ValueError("Audio vide après suppression des silences")

    source_duration_sec = source_duration_ms / 1000.0
    target_duration_sec = target_duration_ms / 1000.0

    # ---------------------------------------------------------
    # Facteur nécessaire
    #
    # Exemple:
    # TTS 4s / cible 2s = 2.0x
    # TTS 2s / cible 4s = 0.5x
    # ---------------------------------------------------------
    required_speed = source_duration_sec / target_duration_sec

    print(
        f"    TTS      : {source_duration_sec:.2f}s"
    )
    print(
        f"    Cible    : {target_duration_sec:.2f}s"
    )
    print(
        f"    Facteur  : {required_speed:.2f}x"
    )

    # ---------------------------------------------------------
    # Limitation du débit
    # ---------------------------------------------------------
    speed = max(
        min_speed,
        min(max_speed, required_speed)
    )

    if abs(speed - required_speed) > 0.01:
        print(
            f"    ⚠️ Ajustement limité : "
            f"{required_speed:.2f}x → {speed:.2f}x"
        )

    # ---------------------------------------------------------
    # Cas proche de la durée cible :
    # pas besoin de modification agressive
    # ---------------------------------------------------------
    if abs(required_speed - 1.0) <= 0.03:
        speed = 1.0
        print("    ✅ Durée suffisamment proche, aucun time-stretch.")

    # ---------------------------------------------------------
    # Préparer fichier temporaire
    # ---------------------------------------------------------
    temp_input = output_path.with_name(
        output_path.stem + "_temp.wav"
    )

    audio.export(
        temp_input,
        format="wav"
    )

    try:
        # -----------------------------------------------------
        # Construire atempo
        # -----------------------------------------------------
        filters = []
        remaining_speed = speed

        while remaining_speed > 2.0:
            filters.append("atempo=2.0")
            remaining_speed /= 2.0

        while remaining_speed < 0.5:
            filters.append("atempo=0.5")
            remaining_speed /= 0.5

        filters.append(
            f"atempo={remaining_speed:.6f}"
        )

        filter_string = ",".join(filters)

        # -----------------------------------------------------
        # FFmpeg
        # -----------------------------------------------------
        command = [
            "ffmpeg",
            "-y",
            "-i", str(temp_input),
            "-filter:a", filter_string,
            "-ar", "24000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(output_path),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg erreur :\n{result.stderr}"
            )

    finally:
        temp_input.unlink(missing_ok=True)

    # ---------------------------------------------------------
    # Durée finale réelle
    # ---------------------------------------------------------
    result_audio = AudioSegment.from_file(output_path)
    final_duration_ms = len(result_audio)

    print(
        f"    ✅ Durée finale : "
        f"{final_duration_ms / 1000:.2f}s"
    )

    return True, final_duration_ms


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
    voice_manager = VoiceManager(
        voices=speaker_references,
        engine=engine,
        gpu=config.USE_CUDA,
        segments_dir=output_dir / "segments"
    )

    dubbed_segments = []
    total = len(dialogues)
    cloned_dir = output_dir / "tts_cloned"

    # Tri des dialogues pour calculer la durée disponible
    dialogues = sorted(dialogues, key=lambda d: (d["start"], d["index"]))

    for idx_loop, dialog in enumerate(dialogues):
        idx = dialog["index"]
        speaker = dialog["speaker"]
        french_text = dialog["french_text"]
        start_sec = dialog["start"]
        end_sec = dialog["end"]
        duration_sec = dialog["duration"]

        # Calcul de la durée réellement disponible jusqu'à la réplique suivante
        guard_sec = float(getattr(config, "DUBBING_GUARD_SECONDS", 0.08))

        if idx_loop + 1 < len(dialogues):
            next_start_sec = dialogues[idx_loop + 1]["start"]
        else:
            next_start_sec = total_duration_ms / 1000.0

        available_sec = next_start_sec - start_sec - guard_sec

        if available_sec <= 0.10:
            print(f"  ⚠️ Aucune place suffisante avant la réplique suivante; réplique {idx} ignorée.")
            continue

        # La durée cible tient compte de la place réelle avant la réplique suivante.
        target_sec = min(duration_sec, available_sec)
        target_duration_ms = target_sec * 1000.0

        print(f"\n[{idx_loop + 1}/{total}] Voix pour {speaker} ({start_sec:.2f}s -> {end_sec:.2f}s, originale {duration_sec:.2f}s, cible {target_sec:.2f}s)")
        print(f"  Texte FR : \"{french_text[:80]}...\" " if len(french_text) > 80 else f"  Texte FR : \"{french_text}\"")

        raw_tts_file = tts_dir / f"tts_{idx:05d}_{speaker}_raw.wav"
        fitted_tts_file = tts_dir / f"tts_{idx:05d}_{speaker}.wav"
        precloned_file = cloned_dir / f"cloned_{idx:05d}_{speaker}.wav"

        try:
            # Fixe une graine aléatoire par acteur pour stabiliser XTTS
            seed = int.from_bytes(speaker.encode("utf-8"), "little") % 2147483647
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # Cache optionnel des fichiers TTS; désactivé par défaut pour
            # éviter de réutiliser une voix générique après un changement de moteur.
            source_file = None
            if config.REUSE_TTS_CACHE and precloned_file.is_file() and precloned_file.stat().st_size > 1000:
                print(f"  ✨ Réutilisation du clonage existant : {precloned_file.name}")
                source_file = precloned_file
            elif config.REUSE_TTS_CACHE and raw_tts_file.is_file() and raw_tts_file.stat().st_size > 1000:
                print(f"  ✨ Réutilisation de l'audio TTS existant : {raw_tts_file.name}")
                source_file = raw_tts_file
            else:
                # Référence audio : toujours priorité à la référence consolidée de l'acteur
                ref_segment = speaker_references.get(speaker)

                if not ref_segment:
                    # Fallback uniquement en secours
                    ref_segment = dialog.get("audio_segment_original")

                if not ref_segment:
                    print(f"  ❌ Aucune référence disponible pour {speaker}, réplique ignorée.")
                    continue

                # Enregistre la voix pour le moteur TTS
                voice_manager.register_voice(speaker, ref_segment)

                # Synthèse vocale avec clonage de l'acteur
                voice_manager.speak(
                    speaker=speaker,
                    text=french_text,
                    output_file=raw_tts_file,
                    language=config.TARGET_LANGUAGE,
                    reference_wav=ref_segment
                )
                source_file = raw_tts_file

            # 2. Ajustement de la durée pour caler au timing exact du film
            success, real_duration_ms = fit_audio_duration(
                source_file,
                fitted_tts_file,
                target_duration_ms,
                min_speed=config.TTS_MIN_SPEED,
                max_speed=config.TTS_MAX_SPEED,
            )

            if not success:
                raise RuntimeError(
                    f"Impossible d'ajuster la réplique {idx}"
                )

            dubbed_segments.append({
                "index": idx,
                "speaker": speaker,
                "start_ms": int(start_sec * 1000),
                "duration_ms": real_duration_ms,
                "target_duration_ms": int(target_duration_ms),
                "end_limit_ms": int(
                    min(
                        start_sec + target_sec,
                        next_start_sec - guard_sec,
                    )
                    * 1000.0
                ),
                "audio_path": str(fitted_tts_file),
            })
            print(f"  ✓ Réplique française synchronisée et calée.")

        except Exception as e:
            print(f"  ❌ Erreur génération réplique {idx} ({speaker}): {e}")

    # 3. Réassemblage de la piste globale synchronisée (stricte sans chevauchement)
    print(f"\nAssemblage final de la piste audio française ({len(dubbed_segments)} répliques assemblées)...")
    final_audio = AudioSegment.silent(duration=total_duration_ms, frame_rate=24000)

    dubbed_segments = sorted(dubbed_segments, key=lambda item: (item["start_ms"], item["index"]))
    for seg in dubbed_segments:
        seg_audio_file = seg["audio_path"]

        if not os.path.exists(seg_audio_file):
            continue

        desired_start_ms = seg["start_ms"]

        # Les répliques gardent leur position source; leur durée est bornée
        # par le début prévu de la suivante afin d'éviter un décalage en chaîne.
        start_ms = desired_start_ms

        if start_ms >= total_duration_ms:
            break

        hard_limit_ms = min(total_duration_ms, seg["end_limit_ms"])

        max_duration_ms = hard_limit_ms - start_ms

        if max_duration_ms < 100:
            continue

        speech = AudioSegment.from_file(seg_audio_file)

        if len(speech) == 0:
            continue

        # Coupe stricte pour éviter tout chevauchement
        if len(speech) > max_duration_ms:
            overflow_ms = len(speech) - max_duration_ms
            speech = speech[:max_duration_ms]

            if overflow_ms > 150:
                print(
                    f"  ⚠️ Réplique {seg['index']} tronquée de {overflow_ms / 1000:.2f}s "
                    f"pour éviter un chevauchement."
                )

        if len(speech) < 80:
            continue

        fade_in_ms = min(15, len(speech) // 4)
        fade_out_ms = min(30, len(speech) // 4)

        speech = speech.fade_in(fade_in_ms).fade_out(fade_out_ms)

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
