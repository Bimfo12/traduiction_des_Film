#!/usr/bin/env python3
"""
Traduction automatique en français (VF) de tous les fichiers de voix
présents dans un dossier (ex: workspace/video/voices) en utilisant la méthode
de main_via_gpt.py (Whisper -> MarianMT -> Google TTS).
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from gtts import gTTS
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer


def natural_sort_key(file_path: Path):
    """Trie les fichiers de manière naturelle (ex: V1, V2, V10...)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file_path.name)]


def traduire_dossier_voix(
    voices_dir: str | Path = "/home/edouard/Bureau/traduiction_des_Film/workspace/video/voices",
    output_dir: str | Path | None = None,
    whisper_model_size: str = "small",
    device: str = "cpu"
):
    voices_dir = Path(voices_dir).resolve()
    if not voices_dir.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {voices_dir}")

    if output_dir is None:
        output_dir = voices_dir.parent / "voices_vf"
    else:
        output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Récupérer tous les fichiers audio du dossier
    audio_files = sorted(
        [f for f in voices_dir.iterdir() if f.is_file() and f.suffix.lower() in [".wav", ".mp3", ".ogg", ".flac", ".m4a"]],
        key=natural_sort_key
    )

    if not audio_files:
        print(f"❌ Aucun fichier audio trouvé dans {voices_dir}")
        return

    print("=" * 75)
    print("🎙️ TRADUCTION DES VOIX DES ACTEURS EN FRANÇAIS (VF) VIA MAIN_VIA_GPT")
    print("=" * 75)
    print(f"Dossier source  : {voices_dir}")
    print(f"Dossier cible   : {output_dir}")
    print(f"Nombre de voix  : {len(audio_files)}")
    print(f"Moteur TTS      : Google Text-to-Speech (gTTS)")
    print("=" * 75)

    # 1. Chargement de Whisper
    print("\n1. Chargement de Whisper...")
    whisper = WhisperModel(whisper_model_size, device=device, compute_type="int8")
    print("✓ Whisper chargé.")

    # 2. Chargement de MarianMT
    print("\n2. Chargement du modèle de traduction MarianMT (en -> fr)...")
    model_name = "Helsinki-NLP/opus-mt-en-fr"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    translation_model = MarianMTModel.from_pretrained(model_name)
    print("✓ MarianMT chargé.")

    recapitulatif = []
    total = len(audio_files)

    print("\n3. Traitement de chaque voix...")
    for idx, voice_file in enumerate(audio_files, start=1):
        voice_name = voice_file.stem
        output_mp3 = output_dir / f"{voice_name}_vf.mp3"
        output_wav = output_dir / f"{voice_name}_vf.wav"

        print(f"\n[{idx}/{total}] Analyse de {voice_file.name}...")

        # A. Transcription avec Whisper
        try:
            segments, _ = whisper.transcribe(
                str(voice_file),
                language="en",
                beam_size=5,
                vad_filter=True
            )
            english_parts = [s.text.strip() for s in segments if s.text.strip()]
            english_text = " ".join(english_parts).strip()
        except Exception as e:
            print(f"  ⚠️ Erreur transcription : {e}")
            english_text = ""

        if not english_text:
            print("  ⚠️ Aucune parole détectée dans cet extrait (bruit ou silence).")
            recapitulatif.append({
                "source_file": voice_file.name,
                "english": "",
                "french": "",
                "status": "aucun_dialogue"
            })
            continue

        print(f"  🇺🇸 Anglais : \"{english_text}\"")

        # B. Traduction avec MarianMT
        try:
            inputs = tokenizer([english_text], return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = translation_model.generate(**inputs, max_length=512)
            french_text = tokenizer.decode(translated[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"  ⚠️ Erreur traduction : {e}")
            french_text = ""

        if not french_text:
            continue

        print(f"  🇫🇷 Français : \"{french_text}\"")

        # C. Génération de la voix française avec gTTS (comme dans main_via_gpt.py)
        try:
            tts = gTTS(text=french_text, lang="fr", slow=False)
            tts.save(str(output_mp3))

            # Conversion en WAV pour compatibilité maximale
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(output_mp3), "-ar", "24000", "-ac", "1", str(output_wav)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )

            print(f"  ✓ Audio VF généré : {output_mp3.name} & {output_wav.name}")
            recapitulatif.append({
                "source_file": voice_file.name,
                "english": english_text,
                "french": french_text,
                "output_mp3": str(output_mp3),
                "output_wav": str(output_wav),
                "status": "succes"
            })
        except Exception as e:
            print(f"  ❌ Erreur génération audio gTTS : {e}")
            recapitulatif.append({
                "source_file": voice_file.name,
                "english": english_text,
                "french": french_text,
                "status": f"erreur_tts: {e}"
            })

    # 4. Sauvegarde du récapitulatif
    json_path = output_dir / "recapitulatif_traduction.json"
    txt_path = output_dir / "recapitulatif_traduction.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recapitulatif, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write(f"RÉCAPITULATIF DE LA TRADUCTION DES VOIX EN VF ({len(recapitulatif)} fichiers)\n")
        f.write("=" * 75 + "\n\n")
        for item in recapitulatif:
            f.write(f"Fichier : {item['source_file']}\n")
            f.write(f"  EN : {item.get('english', '')}\n")
            f.write(f"  FR : {item.get('french', '')}\n")
            f.write(f"  Statut : {item.get('status', '')}\n\n")

    print("\n" + "=" * 75)
    print("🎉 TRADUCTION DE TOUTES LES VOIX TERMINÉE !")
    print("=" * 75)
    print(f"📁 Fichiers VF enregistrés dans : {output_dir}")
    print(f"📄 Récapitulatif : {txt_path}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="Traduit tous les fichiers vocaux d'un dossier en français (VF) avec la méthode main_via_gpt.py."
    )
    parser.add_argument(
        "--voices-dir",
        type=str,
        default="/home/edouard/Bureau/traduiction_des_Film/workspace/video/voices",
        help="Dossier contenant les fichiers audio de voix (défaut: workspace/video/voices)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Dossier où enregistrer les voix françaises (défaut: workspace/video/voices_vf)"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="small",
        help="Taille du modèle Whisper (tiny, base, small, medium)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu ou cuda"
    )
    args = parser.parse_args()

    traduire_dossier_voix(
        voices_dir=args.voices_dir,
        output_dir=args.output_dir,
        whisper_model_size=args.whisper_model,
        device=args.device
    )


if __name__ == "__main__":
    main()

