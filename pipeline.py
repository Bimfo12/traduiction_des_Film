"""
Orchestrateur global du pipeline de traduction et doublage de film entier.

Enchaîne les étapes :
1. Extraction de l'audio du film (step1_extract_audio)
1b. Séparation voix / musique de fond avec Demucs pour préserver l'ambiance et la musique (audio_separator)
2. Diarisation des acteurs et extraction des voix de référence (step2_diarization)
3. Transcription Whisper et Traduction MarianMT (step3_translation)
4. Synthèse vocale française gTTS / XTTS (step4_voice_cloning)
4b. Mixage de la musique de fond originale avec les nouvelles voix françaises (audio_separator)
5. Fusion de la vidéo avec la bande-son finale mixée (step5_merge_video)
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import config
from modules import (
    extraire_audio,
    separer_voix_musique,
    mixer_musique_et_voix,
    separer_acteurs,
    transcrire_et_traduire,
    generer_doublage,
    fusionner_video_audio,
)


class FilmTranslationPipeline:
    """Pipeline complet de traduction et doublage de film entier."""

    def __init__(
        self,
        video_path: str | Path,
        output_video_path: str | Path | None = None,
        work_dir: str | Path | None = None,
        num_speakers: int | None = config.NUMBER_OF_SPEAKERS,
        whisper_model: str = config.WHISPER_MODEL,
        device: str = config.DEVICE,
        tts_engine: str = config.TTS_ENGINE,
        preserve_music: bool = True
    ):
        self.video_path = Path(video_path).resolve()
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Fichier vidéo introuvable : {self.video_path}")

        film_id = self.video_path.stem
        if work_dir is None:
            self.work_dir = config.DEFAULT_WORK_DIR / film_id
        else:
            self.work_dir = Path(work_dir).resolve() / film_id

        self.work_dir.mkdir(parents=True, exist_ok=True)

        if output_video_path is None:
            output_dir = config.DEFAULT_OUTPUT_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            self.output_video_path = output_dir / f"{film_id}_VF.mp4"
        else:
            self.output_video_path = Path(output_video_path).resolve()
            self.output_video_path.parent.mkdir(parents=True, exist_ok=True)

        self.num_speakers = num_speakers
        self.whisper_model = whisper_model
        self.device = device
        self.tts_engine = tts_engine
        self.preserve_music = preserve_music

        # Fichiers intermédiaires clés
        self.extracted_audio = self.work_dir / "audio_original.wav"
        self.vocals_audio = self.work_dir / "vocals.wav"
        self.instrumental_audio = self.work_dir / "no_vocals.wav"
        self.segments_json = self.work_dir / "segments_info.json"
        self.speakers_json = self.work_dir / "speakers_info.json"
        self.dialogues_json = self.work_dir / "dialogues_traduits.json"
        self.french_voices_only = self.work_dir / "french_voices_only.wav"
        self.dubbed_audio = self.work_dir / "film_francais.wav"

    def run_all(self, start_step: int = 1) -> Path:
        start_time = time.time()
        print("\n" + "=" * 80)
        print("🚀 DÉMARRAGE DU PIPELINE DE TRADUCTION DE FILM COMPLET")
        print("=" * 80)
        print(f"🎬 Vidéo source         : {self.video_path}")
        print(f"📁 Dossier de travail    : {self.work_dir}")
        print(f"🎯 Vidéo de sortie      : {self.output_video_path}")
        print(f"⚙️ Périphérique         : {self.device}")
        print(f"🎙️ Moteur TTS           : {self.tts_engine.upper()}")
        print(f"🎵 Conserver la musique : {'OUI (Demucs)' if self.preserve_music else 'NON'}")
        print(f"Étape de départ         : Étape {start_step}")
        print("=" * 80 + "\n")

        # ----------------------------------------------------
        # Étape 1 : Extraction Audio du film
        # ----------------------------------------------------
        if start_step <= 1 or not self.extracted_audio.exists():
            print("\n>>> LANCEMENT ÉTAPE 1/5...")
            extraire_audio(
                video_path=self.video_path,
                audio_path=self.extracted_audio,
                sample_rate=config.AUDIO_EXTRACT_SAMPLE_RATE,
                channels=config.AUDIO_EXTRACT_CHANNELS
            )
        else:
            print(f"\n⏩ Étape 1 ignorée (fichier existant: {self.extracted_audio})")

        # ----------------------------------------------------
        # Étape 1b : Séparation Voix et Musique de fond (Demucs)
        # ----------------------------------------------------
        audio_for_diarization = self.extracted_audio
        if self.preserve_music:
            if not self.vocals_audio.exists() or not self.instrumental_audio.exists():
                print("\n>>> SÉPARATION VOIX / MUSIQUE DE FOND (Demucs)...")
                try:
                    separer_voix_musique(
                        audio_path=self.extracted_audio,
                        output_dir=self.work_dir,
                        device=self.device
                    )
                except Exception as e:
                    print(f"⚠️ Erreur lors de la séparation Demucs ({e}). Poursuite sur l'audio complet.")
            if self.vocals_audio.exists():
                audio_for_diarization = self.vocals_audio

        # ----------------------------------------------------
        # Étape 2 : Diarisation & Voix de Référence
        # ----------------------------------------------------
        if start_step <= 2 or not self.segments_json.exists():
            print("\n>>> LANCEMENT ÉTAPE 2/5...")
            segments_data, speaker_references = separer_acteurs(
                audio_path=audio_for_diarization,
                output_dir=self.work_dir,
                device=self.device,
                number_of_speakers=self.num_speakers
            )
        else:
            print(f"\n⏩ Étape 2 ignorée (fichiers existants dans {self.work_dir})")
            with open(self.segments_json, "r", encoding="utf-8") as f:
                segments_data = json.load(f)
            with open(self.speakers_json, "r", encoding="utf-8") as f:
                speaker_references = json.load(f)["references"]

        # ----------------------------------------------------
        # Étape 3 : Transcription & Traduction
        # ----------------------------------------------------
        if start_step <= 3 or not self.dialogues_json.exists():
            print("\n>>> LANCEMENT ÉTAPE 3/5...")
            dialogues = transcrire_et_traduire(
                segments_data=segments_data,
                output_dir=self.work_dir,
                device=self.device,
                whisper_model_size=self.whisper_model,
                source_lang=config.SOURCE_LANGUAGE,
                target_lang=config.TARGET_LANGUAGE
            )
        else:
            print(f"\n⏩ Étape 3 ignorée (fichier existant: {self.dialogues_json})")
            with open(self.dialogues_json, "r", encoding="utf-8") as f:
                dialogues = json.load(f)

        # ----------------------------------------------------
        # Étape 4 : Synthèse vocale & Assemblage de la voix VF
        # ----------------------------------------------------
        voice_track_target = self.french_voices_only if self.instrumental_audio.exists() else self.dubbed_audio
        if start_step <= 4 or not voice_track_target.exists():
            print("\n>>> LANCEMENT ÉTAPE 4/5...")
            generer_doublage(
                dialogues=dialogues,
                speaker_references=speaker_references,
                original_audio_path=self.extracted_audio,
                output_dir=self.work_dir,
                final_audio_path=voice_track_target,
                engine=self.tts_engine
            )
        else:
            print(f"\n⏩ Étape 4 ignorée (fichier existant: {voice_track_target})")

        # ----------------------------------------------------
        # Étape 4b : Mixage Musique de Fond + Nouvelles Voix VF
        # ----------------------------------------------------
        if self.preserve_music and self.instrumental_audio.exists():
            print("\n>>> MIXAGE DE LA MUSIQUE ORIGINALE AVEC LA VF...")
            mixer_musique_et_voix(
                instrumental_file=self.instrumental_audio,
                voice_file=self.french_voices_only,
                output_file=self.dubbed_audio
            )

        # ----------------------------------------------------
        # Étape 5 : Fusion Vidéo + Bande Son Finale
        # ----------------------------------------------------
        print("\n>>> LANCEMENT ÉTAPE 5/5...")
        final_video = fusionner_video_audio(
            video_path=self.video_path,
            audio_path=self.dubbed_audio,
            output_path=self.output_video_path
        )

        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("🏆 PIPELINE TERMINÉ AVEC SUCCÈS !")
        print("=" * 80)
        print(f"⏱️ Durée totale de traitement : {total_time / 60:.2f} minutes")
        print(f"🎬 Vidéo finale traduite et doublée : {final_video}")
        print("=" * 80 + "\n")
        return final_video


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline automatisé de traduction de film avec clonage vocal et préservation de la musique de fond."
    )
    parser.add_argument(
        "video",
        type=str,
        help="Chemin du fichier vidéo à traduire (ex: film.mp4)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Chemin de la vidéo finale de sortie (ex: output/film_vf.mp4)"
    )
    parser.add_argument(
        "-w", "--work-dir",
        type=str,
        default=None,
        help="Dossier de travail pour stocker les fichiers temporaires"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Étape de démarrage (1=tout, 2=diarisation, 3=traduction, 4=doublage, 5=fusion)"
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="Nombre connu d'acteurs (si None, détection automatique par clustering)"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=config.WHISPER_MODEL,
        help="Modèle Faster-Whisper à utiliser (tiny, base, small, medium, large-v3)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        help="Périphérique d'exécution ('cuda' ou 'cpu')"
    )
    parser.add_argument(
        "--tts",
        type=str,
        default=config.TTS_ENGINE,
        choices=["gtts", "xtts"],
        help="Moteur TTS : 'gtts' (Google TTS comme main_via_gpt.py) ou 'xtts' (clonage vocal)"
    )
    parser.add_argument(
        "--no-music",
        action="store_true",
        help="Désactiver la séparation et conservation de la musique de fond"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = FilmTranslationPipeline(
        video_path=args.video,
        output_video_path=args.output,
        work_dir=args.work_dir,
        num_speakers=args.speakers,
        whisper_model=args.whisper_model,
        device=args.device,
        tts_engine=args.tts,
        preserve_music=not args.no_music
    )
    pipeline.run_all(start_step=args.step)
