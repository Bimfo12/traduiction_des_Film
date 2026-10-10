"""
Étape 3 : Transcription audio et Préparation de la Traduction (Malgache / Manuelle ou Automatique).

Fonctionnalités :
1. Transcription automatique avec Faster-Whisper : supporte les vidéos en Français ou en Anglais
   (détection automatique de la langue avec 'auto').
2. Traduction en Malgache par complétion manuelle :
   - Génère un fichier texte clair 'dialogues_malgache.txt' avec les répliques en malgache laissées VIDES (MG: ).
   - L'utilisateur peut écouter chaque segment et remplir ses traductions directement dans le fichier texte.
   - Synchronisation automatique entre 'dialogues_malgache.txt' et 'dialogues_traduits.json'.
3. Poursuite du pipeline sur commande terminal une fois le texte complété.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Permet l'import direct de config quel que soit le dossier d'exécution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from faster_whisper import WhisperModel

import config


def _estimer_duree_paroles(texte: str, caracteres_par_seconde: float = 14.0) -> float:
    """
    Estimation de la durée de parole en secondes.
    Utile pour avertir si une réplique risque d'être trop longue.
    """
    if not texte:
        return 0.0

    propre = "".join(ch for ch in texte if ch.isalnum() or ch.isspace())
    return max(0.30, len(propre) / caracteres_par_seconde)


def generer_fichier_texte_malgache(dialogues: List[Dict], txt_path: Path) -> Path:
    """
    Génère un fichier texte propre et lisible pour la traduction en malgache.
    Laisse la ligne 'MG: ' vide pour que l'utilisateur puisse la compléter manuellement,
    ou préserve le texte déjà saisi si disponible.
    """
    txt_path = Path(txt_path).resolve()
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    lignes = [
        "# ==============================================================================",
        "# 🇲🇬 FICHIER DE TRADUCTION EN MALGACHE DES RÉPLIQUES DU FILM",
        "# ==============================================================================",
        "# INSTRUCTIONS :",
        "# 1. Complétez votre traduction en malgache sur la ligne 'MG: ' pour chaque réplique.",
        "# 2. Vous pouvez écouter l'audio original dans le dossier 'segments/' de votre workspace.",
        "# 3. Ne modifiez pas les lignes d'en-tête [REPLIQUE ...] ni 'ORIGINAL:'.",
        "# 4. Une fois terminé, enregistrez ce fichier et lancez la commande suivante :",
        "#      python main.py <nom_video> --continuer",
        "#    (ou : python pipeline.py <nom_video> --step 4)",
        "# ==============================================================================\n",
    ]

    for d in dialogues:
        idx = d["index"]
        speaker = d["speaker"]
        start = d["start"]
        end = d["end"]
        duration = d["duration"]
        orig_text = d.get("original_text", "").strip()
        mg_text = d.get("malgache_text") or d.get("target_text") or ""

        # Nom du fichier audio de la réplique
        audio_ref = Path(d.get("audio_segment_original", "")).name if d.get("audio_segment_original") else ""

        lignes.append(f"[REPLIQUE {idx:05d}] [{start:06.2f}s -> {end:06.2f}s] (Durée: {duration:.2f}s) Acteur: {speaker}")
        if audio_ref:
            lignes.append(f"AUDIO: {audio_ref}")
        lignes.append(f"ORIGINAL: {orig_text}")
        lignes.append(f"MG: {mg_text.strip()}\n")

    txt_path.write_text("\n".join(lignes), encoding="utf-8")
    return txt_path


def synchroniser_dialogues_depuis_txt(
    txt_path: Path,
    json_path: Optional[Path] = None
) -> Tuple[List[Dict], int, int]:
    """
    Lit le fichier texte 'dialogues_malgache.txt' complété par l'utilisateur,
    met à jour les dialogues et synchronise 'dialogues_traduits.json'.

    Retourne :
        (dialogues_a_jour, nombre_completes, nombre_vides)
    """
    txt_path = Path(txt_path).resolve()
    if not txt_path.is_file():
        raise FileNotFoundError(f"Fichier de dialogues malgache introuvable : {txt_path}")

    # Charger le JSON existant s'il existe pour conserver les métadonnées audio
    base_dialogues: Dict[int, Dict] = {}
    if json_path and Path(json_path).is_file():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    base_dialogues[item["index"]] = item
        except Exception:
            pass

    content = txt_path.read_text(encoding="utf-8")

    # Regex pour extraire chaque bloc de réplique
    pattern = re.compile(
        r"\[REPLIQUE\s+(\d+)\]"
        r"(?:.*?\[([\d\.]+)s\s*->\s*([\d\.]+)s\])?"
        r"(?:.*?Acteur:\s*([A-Za-z0-9_]+))?"
        r"(?:.*?AUDIO:\s*([^\n]+))?"
        r".*?ORIGINAL:\s*([^\n]*)"
        r".*?MG:\s*([^\n]*)",
        re.DOTALL
    )

    updated_dialogues = []
    nb_completes = 0
    nb_vides = 0

    for match in pattern.finditer(content):
        idx = int(match.group(1))
        start_str = match.group(2)
        end_str = match.group(3)
        speaker_str = match.group(4)
        audio_name = match.group(5)
        orig_text = match.group(6).strip()
        mg_text = match.group(7).strip()

        # Récupérer l'élément de base ou reconstruire
        base = base_dialogues.get(idx, {})
        start = float(start_str) if start_str else base.get("start", 0.0)
        end = float(end_str) if end_str else base.get("end", 0.0)
        duration = round(end - start, 3) if end > start else base.get("duration", 0.0)
        speaker = speaker_str if speaker_str else base.get("speaker", f"V1")

        audio_path = base.get("audio_segment_original")
        if not audio_path and audio_name:
            audio_path = str(txt_path.parent / "segments" / audio_name.strip())

        if mg_text:
            nb_completes += 1
        else:
            nb_vides += 1

        entry = {
            "index": idx,
            "speaker": speaker,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(duration, 3),
            "audio_segment_original": audio_path,
            "original_text": orig_text or base.get("original_text", ""),
            "malgache_text": mg_text,
            "target_text": mg_text,
            "french_text": mg_text,  # Rétrocompatibilité avec les étapes suivantes
        }
        updated_dialogues.append(entry)

    # Trier par start
    updated_dialogues = sorted(updated_dialogues, key=lambda d: (d["start"], d["index"]))

    # Synchroniser dans le JSON si un chemin est fourni
    if json_path:
        json_path = Path(json_path).resolve()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(updated_dialogues, f, ensure_ascii=False, indent=2)

    return updated_dialogues, nb_completes, nb_vides


class TranscripteurTraducteur:
    """
    Gère la transcription audio (Faster-Whisper) et la traduction (manuelle ou automatique).
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
        print("✓ Faster-Whisper chargé avec succès.")

        self.translator = None
        self.tokenizer = None

        # Modèle MarianMT uniquement si traduction automatique demandée vers le français
        if not getattr(config, "MANUAL_TRANSLATION", True) and target_lang == "fr":
            try:
                from transformers import MarianMTModel, MarianTokenizer
                print(f"Chargement du modèle de traduction automatique MarianMT ({translation_model_name})...")
                self.tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
                self.translator = MarianMTModel.from_pretrained(translation_model_name)
                self.torch_device = torch.device(device)
                self.translator.to(self.torch_device)
                print("✓ MarianMT chargé.")
            except Exception as e:
                print(f"⚠️ MarianMT non disponible ({e}), repli sur la traduction manuelle.")

    def transcrire_audio(self, audio_path: str | Path) -> Tuple[str, str]:
        """
        Transcrit un segment audio.
        Détecte automatiquement si la langue parlée est le Français ou l'Anglais
        si source_lang est 'auto'.
        
        Retourne : (texte_transcrit, langue_detectee)
        """
        lang_param = None if (not self.source_lang or self.source_lang.lower() == "auto") else self.source_lang

        segments, info = self.whisper.transcribe(
            str(audio_path),
            language=lang_param,
            beam_size=config.WHISPER_BEAM_SIZE,
            vad_filter=True,
            condition_on_previous_text=False,
            vad_parameters={"min_silence_duration_ms": 350},
        )

        texts = [segment.text.strip() for segment in segments if segment.text.strip()]
        detected_lang = getattr(info, "language", self.source_lang or "inconnu")
        return " ".join(texts).strip(), detected_lang

    def traduire_texte(self, text: str) -> str:
        """
        Traduit automatiquement avec MarianMT si configuré, sinon retourne une chaîne vide.
        """
        if not text or not text.strip() or self.translator is None or self.tokenizer is None:
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
    Parcourt chaque segment audio, transcrit les paroles originales (FR ou EN),
    puis prépare le fichier texte pour la complétion des paroles en malgache.

    Args:
        segments_data: Liste des segments issus de l'étape 2.
        output_dir: Dossier de travail.

    Returns:
        Liste des segments préparés.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "dialogues_traduits.json"
    txt_path = output_dir / getattr(config, "DIALOGUES_MALGACHE_TXT", "dialogues_malgache.txt")

    print("=" * 60)
    print("📝 ÉTAPE 3 : TRANSCRIPTION & PRÉPARATION DE LA TRADUCTION MALGACHE")
    print("=" * 60)
    print(f"Langue source demandée : {source_lang} (détection automatique du Français/Anglais si 'auto')")
    print(f"Langue cible           : Malgache ({target_lang})")
    print(f"Mode de traduction     : Manuel (paroles en malgache laissées vides à compléter)")
    print(f"Segments à traiter     : {len(segments_data)}")

    # Vérifier s'il existe déjà un travail en cours dans le fichier texte pour ne pas l'écraser
    existing_mg_texts: Dict[int, str] = {}
    if txt_path.is_file():
        try:
            prev_dialogues, _, _ = synchroniser_dialogues_depuis_txt(txt_path)
            for d in prev_dialogues:
                if d.get("malgache_text"):
                    existing_mg_texts[d["index"]] = d["malgache_text"]
            if existing_mg_texts:
                print(f"✨ {len(existing_mg_texts)} traduction(s) malgache(s) préexistante(s) conservée(s).")
        except Exception as e:
            print(f"⚠️ Note sur fichier texte existant : {e}")

    trans_engine = TranscripteurTraducteur(
        whisper_model_size=whisper_model_size,
        device=device,
        source_lang=source_lang,
        target_lang=target_lang
    )

    dialogues = []
    total = len(segments_data)

    for i, seg in enumerate(segments_data, start=1):
        idx = seg["index"]
        audio_file = seg["audio_path"]
        speaker = seg["speaker"]
        start = seg["start"]
        end = seg["end"]
        duration = seg["duration"]

        print(f"\n[{i}/{total}] Réplique de {speaker} ({start:.2f}s -> {end:.2f}s, {duration:.2f}s)")

        # 1. Transcription Faster-Whisper
        original_text, lang_detected = trans_engine.transcrire_audio(audio_file)
        if not original_text:
            print("  (Aucune parole détectée, réplique ignorée)")
            continue

        print(f"  🗣️ [{lang_detected.upper()}] : \"{original_text}\"")

        # 2. Texte malgache : récupère l'existant s'il y a déjà une saisie, sinon vide
        malgache_text = existing_mg_texts.get(idx, "")
        if malgache_text:
            print(f"  🇲🇬 [MG déjà renseigné] : \"{malgache_text}\"")
        else:
            print(f"  🇲🇬 [MG laissé vide] : (à compléter dans {txt_path.name})")

        dialogues.append({
            "index": idx,
            "speaker": speaker,
            "start": start,
            "end": end,
            "duration": duration,
            "audio_segment_original": audio_file,
            "original_text": original_text,
            "malgache_text": malgache_text,
            "target_text": malgache_text,
            "french_text": malgache_text,  # Rétrocompatibilité
        })

    # Sauvegarde des dialogues en JSON et en TXT
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=2)

    generer_fichier_texte_malgache(dialogues, txt_path)

    # Affichage des instructions pour l'utilisateur
    print("\n" + "=" * 70)
    print("✅ ÉTAPE 3 TERMINÉE : FICHIER TEXTE EN MALGACHE PRÊT !")
    print("=" * 70)
    print(f"📄 Fichier texte à compléter :")
    print(f"   👉 {txt_path}")
    print("\n📌 PROCÉDURE À SUIVRE :")
    print("  1. Ouvrez le fichier ci-dessus avec votre éditeur de texte.")
    print("  2. Pour chaque réplique, écrivez la traduction en malgache après 'MG: '.")
    print("  3. Enregistrez les modifications du fichier.")
    print("  4. Lancez ensuite la commande suivante dans le terminal pour continuer :")
    print(f"     👉 python main.py <nom_de_la_video> --continuer")
    print(f"     (ou : python pipeline.py <nom_de_la_video> --step 4)")
    print("=" * 70 + "\n")

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
