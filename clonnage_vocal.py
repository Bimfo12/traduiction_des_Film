#!/usr/bin/env python3
"""
Clonage Vocal Amélioré avec XTTS v2.
Basé sur clonnage-vocal/clonnage-vocal.py avec les améliorations suivantes :
1. Utilise directement les extraits audio des acteurs dans 'workspace/video/segments' comme référence vocale (speaker_wav).
2. Patch automatique pour éviter les erreurs de bibliothèques CUDA (torchcodec / libnvrtc.so.13).
3. Permet de cloner la voix d'un acteur spécifique pour n'importe quel texte en français.
4. Permet de cloner et doubler automatiquement toutes les répliques du film avec la vraie voix de chaque acteur.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Permet l'import direct de config quel que soit le dossier d'exécution
sys.path.insert(0, str(Path(__file__).resolve().parent))

import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment

# ---------------------------------------------------------------------------
# Patch de sécurité pour torchaudio.load (évite l'erreur torchcodec / libnvrtc.so.13)
# ---------------------------------------------------------------------------
_orig_torchaudio_load = torchaudio.load

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
        return _orig_torchaudio_load(filepath, *args, **kwargs)

torchaudio.load = _safe_torchaudio_load

from TTS.api import TTS
import config


class CloneurVocalActeur:
    """
    Gestionnaire de clonage vocal XTTS v2 utilisant les segments originaux des acteurs.
    """

    def __init__(
        self,
        segments_dir: str | Path = "workspace/video/segments",
        model_name: str = config.XTTS_MODEL_NAME,
        gpu: bool = config.USE_CUDA
    ):
        self.segments_dir = Path(segments_dir).resolve()
        self.model_name = model_name
        self.use_gpu = gpu and torch.cuda.is_available()

        print("=" * 70)
        print("🎙️ INITIALISATION DU CLONEUR VOCAL XTTS V2")
        print("=" * 70)
        print(f"Modèle XTTS         : {self.model_name}")
        print(f"Périphérique        : {'GPU NVIDIA' if self.use_gpu else 'CPU'}")
        print(f"Dossier des segments: {self.segments_dir}")

        print("Chargement du modèle XTTS v2 en mémoire...")
        self.tts = TTS(
            model_name=self.model_name,
            progress_bar=False,
            gpu=self.use_gpu
        )
        print("✓ Modèle XTTS v2 prêt.")

        self.actor_segments = self._indexer_segments_acteurs()

    def _indexer_segments_acteurs(self) -> Dict[str, List[Path]]:
        """Indexe tous les fichiers de segments par acteur (V1, V2, V3...)."""
        indexed = {}
        if not self.segments_dir.is_dir():
            print(f"⚠️ Le dossier des segments n'existe pas encore : {self.segments_dir}")
            return indexed

        for file in sorted(self.segments_dir.glob("*.wav")):
            # Format attendu : segment_00004_V4.wav
            match = re.search(r"_(V\d+)\.wav$", file.name)
            if match:
                actor = match.group(1)
                indexed.setdefault(actor, []).append(file)
            else:
                # Segment sans nom d'acteur spécifique
                indexed.setdefault("default", []).append(file)

        print(f"✓ {len(indexed)} acteur(s) répertorié(s) dans les segments.")
        return indexed

    def get_reference_audio(self, actor_or_segment: str) -> Path:
        """
        Trouve le meilleur fichier audio de référence pour un acteur.
        Sélectionne le segment le plus long de cet acteur pour maximiser la qualité du clonage.
        """
        # Si c'est un chemin direct vers un fichier
        direct_path = Path(actor_or_segment)
        if direct_path.is_file():
            return direct_path

        candidate_file = self.segments_dir / actor_or_segment
        if candidate_file.is_file():
            return candidate_file

        # Si c'est un identifiant d'acteur comme 'V4' ou 'V2'
        actor_name = actor_or_segment.upper()
        if actor_name in self.actor_segments and self.actor_segments[actor_name]:
            # Trier les segments de cet acteur par taille de fichier (plus long = meilleure référence)
            best_segment = max(self.actor_segments[actor_name], key=lambda f: f.stat().st_size)
            return best_segment

        raise FileNotFoundError(
            f"Impossible de trouver un audio de référence pour '{actor_or_segment}'. "
            f"Acteurs disponibles : {list(self.actor_segments.keys())}"
        )

    def cloner_voix(
        self,
        text: str,
        reference_segment: str | Path,
        output_file: str | Path,
        language: str = "fr"
    ) -> Path:
        """
        Génère une phrase en clonant la voix de l'acteur issu du segment donné.
        """
        ref_path = self.get_reference_audio(reference_segment)
        out_path = Path(output_file).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n🎙️ Génération de parole avec la voix de l'acteur :")
        print(f"  • Référence vocale : {ref_path.name} ({ref_path})")
        print(f"  • Texte FR         : \"{text}\"")
        print(f"  • Fichier cible    : {out_path.name}")

        self.tts.tts_to_file(
            text=text,
            speaker_wav=str(ref_path),
            language=language,
            file_path=str(out_path)
        )

        print(f"✅ Voix générée avec succès : {out_path}")
        return out_path

    def cloner_tous_les_dialogues(
        self,
        dialogues_json: str | Path = "workspace/video/dialogues_traduits.json",
        output_dir: str | Path = "workspace/video/tts_cloned"
    ):
        """
        Clone la voix de chaque acteur pour toutes les répliques traduites du film.
        """
        dialogues_json = Path(dialogues_json).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if not dialogues_json.is_file():
            raise FileNotFoundError(f"Fichier de dialogues introuvable : {dialogues_json}")

        with open(dialogues_json, "r", encoding="utf-8") as f:
            dialogues = json.load(f)

        print("=" * 70)
        print(f"🎬 CLONAGE EN LOT DES {len(dialogues)} RÉPLIQUES DU FILM")
        print("=" * 70)

        for i, d in enumerate(dialogues, start=1):
            idx = d["index"]
            speaker = d["speaker"]
            french_text = d["french_text"]
            original_segment = d.get("audio_segment_original")

            # Référence : soit le segment exact de la réplique, soit le meilleur segment de l'acteur
            ref = original_segment if (original_segment and Path(original_segment).is_file()) else speaker

            out_file = output_dir / f"cloned_{idx:05d}_{speaker}.wav"
            print(f"\n[{i}/{len(dialogues)}] Acteur {speaker} - Réplique #{idx}")

            try:
                self.cloner_voix(
                    text=french_text,
                    reference_segment=ref,
                    output_file=out_file,
                    language="fr"
                )
            except Exception as e:
                print(f"❌ Erreur lors du clonage de la réplique #{idx} : {e}")

        print("\n" + "=" * 70)
        print(f"🎉 Toutes les répliques clonées sont disponibles dans : {output_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Clonage vocal XTTS v2 basé sur les segments originaux des acteurs."
    )
    parser.add_argument(
        "--segments-dir",
        type=str,
        default="/home/edouard/Bureau/traduiction_des_Film/workspace/video/segments",
        help="Dossier contenant les segments audio des acteurs"
    )
    parser.add_argument(
        "--segment",
        type=str,
        default=None,
        help="Chemin ou nom du fichier segment de l'acteur (ex: segment_00004_V4.wav)"
    )
    parser.add_argument(
        "--actor",
        type=str,
        default="V4",
        help="Nom de l'acteur (ex: V4, V2...) si --segment n'est pas fourni"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Bonjour, ceci est un test de synthèse vocale utilisant ma propre voix d'acteur.",
        help="Texte en français à faire prononcer par l'acteur"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sortie_acteur_clone.wav",
        help="Chemin du fichier audio de sortie"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lister tous les acteurs et segments disponibles dans workspace/video/segments"
    )
    parser.add_argument(
        "--all-dialogues",
        action="store_true",
        help="Cloner automatiquement toutes les répliques traduites du film (dialogues_traduits.json)"
    )
    parser.add_argument(
        "--dialogues-json",
        type=str,
        default="/home/edouard/Bureau/traduiction_des_Film/workspace/video/dialogues_traduits.json",
        help="Chemin du fichier dialogues_traduits.json"
    )

    args = parser.parse_args()

    # Si l'utilisateur demande juste de lister les acteurs disponibles
    if args.list:
        seg_dir = Path(args.segments_dir).resolve()
        files = sorted(seg_dir.glob("*.wav"))
        actors = {}
        for f in files:
            m = re.search(r"_(V\d+)\.wav$", f.name)
            if m:
                actors.setdefault(m.group(1), []).append(f)
        print("=" * 60)
        print(f"📁 SEGMENTS TROUVÉS DANS : {seg_dir}")
        print(f"Total de segments : {len(files)} | Acteurs : {len(actors)}")
        print("=" * 60)
        for act, seg_list in sorted(actors.items(), key=lambda x: int(x[0][1:])):
            best = max(seg_list, key=lambda f: f.stat().st_size)
            print(f"  • Acteur {act:5s} : {len(seg_list)} segment(s) | Meilleure référence : {best.name}")
        return

    cloneur = CloneurVocalActeur(segments_dir=args.segments_dir)

    if args.all_dialogues:
        cloneur.cloner_tous_les_dialogues(dialogues_json=args.dialogues_json)
    else:
        ref_source = args.segment if args.segment else args.actor
        cloneur.cloner_voix(
            text=args.text,
            reference_segment=ref_source,
            output_file=args.output,
            language="fr"
        )


if __name__ == "__main__":
    main()

