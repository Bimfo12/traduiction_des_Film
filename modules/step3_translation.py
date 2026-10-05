"""
Étape 3 : Transcription et Traduction.
Basé sur main_avec_voice_cloning.py (Whisper + MarianMT Helsinki-NLP/opus-mt-en-fr)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Permet l'import direct de config quel que soit le dossier d'exécution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer

import config


def _estimer_duree_paroles_fr(texte: str, caracteres_par_seconde: float = 14.0) -> float:
    """
    Estimation grossière de la durée de parole en français.
    Utile pour détecter les traductions trop longues.
    """
    if not texte:
        return 0.0

    propre = "".join(
        ch for ch in texte
        if ch.isalnum() or ch.isspace()
    )

    return max(0.30, len(propre) / caracteres_par_seconde)


class TranscripteurTraducteur:
    """
    Gère la transcription audio (Faster-Whisper) et la traduction (MarianMT).
    """

    def __init__(
        self,
        whisper_model_size: str = config.WHISPER_MODEL,
        translation_model_name: str = config.TRANSLATION_MODEL,
        device: str = config.DEVICE,
        compute_type: str = config.WHISPER_COMPUTE_TYPE,
        source_lang: str = config.SOURCE_LANGUAGE,
        target_lang: str = config.TARGET_LANGUAGE,
    ):
        self.device = device
        self.source_lang = source_lang
        self.target_lang = target_lang

        print("Chargement du modèle de transcription Faster-Whisper...")
        self.whisper = WhisperModel(
            whisper_model_size,
            device=device,
            compute_type=compute_type
        )
        print("✓ Faster-Whisper chargé.")

        print(f"Chargement du modèle de traduction MarianMT ({translation_model_name})...")
        self.tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
        self.translator = MarianMTModel.from_pretrained(translation_model_name)

        self.torch_device = torch.device(device)
        self.translator.to(self.torch_device)
        print("✓ MarianMT chargé.")

    def transcrire_audio(self, audio_path: str | Path) -> str:
        """
        Transcrit un segment audio en texte source (ex: anglais).
        """
        segments, _ = self.whisper.transcribe(
            str(audio_path),
            language=self.source_lang,
            beam_size=config.WHISPER_BEAM_SIZE,
            vad_filter=True,
            condition_on_previous_text=False,
            vad_parameters={"min_silence_duration_ms": 350},
        )

        texts = [segment.text.strip() for segment in segments if segment.text.strip()]
        return " ".join(texts).strip()

    def traduire_texte(self, text: str) -> str:
        """
        Traduit une phrase vers la langue cible (ex: français) avec MarianMT.
        """
        if not text or not text.strip():
            return ""

        inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(self.torch_device) for key, value in inputs.items()}

        with torch.no_grad():
            translated = self.translator.generate(**inputs, max_length=512)

        result = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        return result.strip()


def transcrire_et_traduire(
    segments_data: List[Dict],
    output_dir: str | Path,
    device: str = config.DEVICE,
    whisper_model_size: str = config.WHISPER_MODEL,
    source_lang: str = config.SOURCE_LANGUAGE,
    target_lang: str = config.TARGET_LANGUAGE,
) -> List[Dict]:
    """
    Parcourt chaque segment audio découpé, le transcrit puis le traduit.

    Args:
        segments_data: Liste des segments issus de l'étape 2.
        output_dir: Dossier où enregistrer les dialogues traduits.

    Returns:
        Liste des segments enrichis avec 'original_text' et 'french_text'.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📝 ÉTAPE 3 : TRANSCRIPTION & TRADUCTION DES RÉPLIQUES")
    print("=" * 60)
    print(f"Langue source : {source_lang} -> Langue cible : {target_lang}")
    print(f"Segments à traiter : {len(segments_data)}")

    trans_engine = TranscripteurTraducteur(
        whisper_model_size=whisper_model_size,
        device=device,
        source_lang=source_lang,
        target_lang=target_lang
    )

    dialogues = []
    total = len(segments_data)

    for i, seg in enumerate(segments_data, start=1):
        audio_file = seg["audio_path"]
        speaker = seg["speaker"]
        start = seg["start"]
        end = seg["end"]
        duration = seg["duration"]

        print(f"\n[{i}/{total}] Réplique de {speaker} ({start:.2f}s -> {end:.2f}s, {duration:.2f}s)")

        # 1. Transcription Whisper
        original_text = trans_engine.transcrire_audio(audio_file)
        if not original_text:
            print("  (Aucun texte détecté, réplique ignorée)")
            continue

        print(f"  🇺🇸 {original_text}")

        # 2. Traduction MarianMT
        french_text = trans_engine.traduire_texte(original_text)
        print(f"  🇫🇷 {french_text}")

        # 3. Vérifier si la traduction tentra dans la durée disponible
        estimated_duration = _estimer_duree_paroles_fr(
            french_text,
            caracteres_par_seconde=float(getattr(config, "TTS_CHARS_PER_SEC", 14.0))
        )
        if estimated_duration > duration * 1.25:
            print(f"  ⚠️ Traduction probablement trop longue : estimée {estimated_duration:.2f}s pour {duration:.2f}s.")

        dialogues.append({
            "index": seg["index"],
            "speaker": speaker,
            "start": start,
            "end": end,
            "duration": duration,
            "audio_segment_original": audio_file,
            "original_text": original_text,
            "french_text": french_text
        })

    # Sauvegarde des dialogues
    json_path = output_dir / "dialogues_traduits.json"
    txt_path = output_dir / "dialogues_traduits.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== DIALOGUES TRADUITS ({len(dialogues)} répliques) ===\n\n")
        for d in dialogues:
            f.write(f"[{d['start']:06.2f}s -> {d['end']:06.2f}s] {d['speaker']}:\n")
            f.write(f"  EN: {d['original_text']}\n")
            f.write(f"  FR: {d['french_text']}\n\n")

    print(f"\n✅ Transcription et traduction terminées : {len(dialogues)} répliques prêtes pour le doublage.")
    print(f"Fichiers enregistrés : {json_path} et {txt_path}")
    return dialogues


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python step3_translation.py <chemin_segments_info.json> [dossier_sortie]")
        sys.exit(1)

    json_in = Path(sys.argv[1])
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else json_in.parent

    with open(json_in, "r", encoding="utf-8") as f_in:
        segments_input = json.load(f_in)

    transcrire_et_traduire(segments_input, out_dir)
