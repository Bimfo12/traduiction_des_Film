"""
Étape 2 : Diarisation des locuteurs et extraction des références vocales des acteurs.
Basé sur la logique de diarisation de traductionFilm/dubbing.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Permet l'import direct de config quel que soit le dossier d'exécution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torchaudio
import soundfile as sf
from pydub import AudioSegment
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier

import config


def separer_acteurs(
    audio_path: str | Path,
    output_dir: str | Path,
    device: str = config.DEVICE,
    energy_threshold: float = config.ENERGY_THRESHOLD,
    cluster_distance_threshold: float = config.CLUSTER_DISTANCE_THRESHOLD,
    number_of_speakers: int | None = config.NUMBER_OF_SPEAKERS,
    reference_max_seconds: float = config.REFERENCE_MAX_SECONDS,
    merge_gap_seconds: float = config.MERGE_GAP_SECONDS,
    min_segment_seconds: float = config.MIN_SEGMENT_SECONDS,
) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Détecte les acteurs dans la piste audio, découpe les segments parlés,
    et extrait pour chaque acteur un échantillon vocal de référence pour le clonage.

    Args:
        audio_path: Chemin du fichier audio complet.
        output_dir: Dossier de travail pour enregistrer les segments et références.

    Returns:
        Tuple (segments_data, speaker_references):
            - segments_data: Liste de dictionnaires décrivant chaque réplique
            - speaker_references: Dictionnaire { 'V1': '/chemin/vers/V1_reference.wav', ... }
    """
    audio_path = Path(audio_path).resolve()
    output_dir = Path(output_dir).resolve()

    segments_dir = output_dir / "segments"
    voices_dir = output_dir / "voices"
    segments_dir.mkdir(parents=True, exist_ok=True)
    voices_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("👥 ÉTAPE 2 : SÉPARATION DES ACTEURS & EXTRACTION DES VOIX")
    print("=" * 60)
    print(f"Fichier audio : {audio_path}")
    print(f"Périphérique  : {device}")

    # 1. Chargement de l'audio complet avec pydub pour les découpes précises
    print("Chargement de l'audio complet...")
    original_audio = AudioSegment.from_file(str(audio_path))
    audio_duration_sec = len(original_audio) / 1000.0
    print(f"Durée totale : {audio_duration_sec:.2f} secondes ({audio_duration_sec / 60:.2f} minutes)")

    # 2. Préparation du modèle SpeechBrain
    print("Chargement du modèle d'empreinte vocale SpeechBrain...")
    pretrained_cache = config.BASE_DIR / "pretrained_models" / "spkrec-ecapa-voxceleb"
    speaker_model = EncoderClassifier.from_hparams(
        source=config.SPEECHBRAIN_MODEL_SOURCE,
        savedir=str(pretrained_cache),
        run_opts={"device": device}
    )

    # 3. Chargement de l'audio en 16 kHz mono pour l'analyse (via soundfile pour éviter les dépendances CUDA)
    print("Analyse du signal audio (16 kHz mono)...")
    data, sample_rate = sf.read(str(audio_path), dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.transpose(0, 1)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != config.SPEECHBRAIN_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=config.SPEECHBRAIN_SAMPLE_RATE
        )
        waveform = resampler(waveform)
        sample_rate = config.SPEECHBRAIN_SAMPLE_RATE

    waveform = waveform.squeeze(0)
    window_size = int(config.WINDOW_SECONDS * sample_rate)
    step_size = int(config.STEP_SECONDS * sample_rate)

    # 4. Recherche des fenêtres d'énergie vocale
    print("Recherche des fenêtres de dialogue...")
    speech_windows = []
    position = 0

    while position + window_size <= waveform.numel():
        chunk = waveform[position:position + window_size]
        energy = torch.sqrt(torch.mean(chunk ** 2)).item()

        if energy > energy_threshold:
            speech_windows.append({
                "start": position / sample_rate,
                "end": (position + window_size) / sample_rate,
                "audio": chunk
            })

        position += step_size

    print(f"Fenêtres de parole détectées : {len(speech_windows)}")
    if not speech_windows:
        raise RuntimeError(
            "Aucun passage parlé détecté. Vérifiez le volume audio ou ajustez ENERGY_THRESHOLD."
        )

    # 5. Extraction des embeddings vocaux
    print("Extraction des empreintes vocales des acteurs...")
    embeddings = []
    total_windows = len(speech_windows)
    for idx, item in enumerate(speech_windows, start=1):
        if idx % 20 == 0 or idx == total_windows:
            print(f"\rTraitement empreintes : {idx}/{total_windows} ({idx * 100 // total_windows}%)", end="", flush=True)

        audio_batch = item["audio"].unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = speaker_model.encode_batch(audio_batch)
        embeddings.append(embedding.squeeze().detach().cpu().numpy())

    print("\nExtraction des empreintes terminée.")
    X = np.asarray(embeddings)

    # 6. Clustering des locuteurs
    print("Regroupement et identification des acteurs...")
    if number_of_speakers is not None and number_of_speakers > 0:
        speaker_count = min(number_of_speakers, len(X))
        clustering = AgglomerativeClustering(
            n_clusters=speaker_count,
            metric="cosine",
            linkage="average"
        )
        labels = clustering.fit_predict(X)
    elif len(X) == 1:
        labels = np.zeros(1, dtype=int)
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cluster_distance_threshold,
            metric="cosine",
            linkage="average"
        )
        labels = clustering.fit_predict(X)

    raw_segments = []
    for item, label in zip(speech_windows, labels):
        raw_segments.append({
            "start": item["start"],
            "end": item["end"],
            "speaker_id": int(label)
        })

    # 7. Fusion des segments adjacents du même acteur
    merged_segments = []
    for segment in raw_segments:
        if not merged_segments:
            merged_segments.append(segment.copy())
            continue

        previous = merged_segments[-1]
        same_speaker = previous["speaker_id"] == segment["speaker_id"]
        gap = segment["start"] - previous["end"]

        if same_speaker and gap <= merge_gap_seconds:
            previous["end"] = max(previous["end"], segment["end"])
        else:
            merged_segments.append(segment.copy())

    # Filtrer les segments trop courts
    merged_segments = [s for s in merged_segments if (s["end"] - s["start"]) >= min_segment_seconds]

    # Attribution de noms conviviaux aux acteurs (V1, V2, ...) selon leur ordre d'apparition
    speaker_mapping = {}
    for segment in merged_segments:
        spk_id = segment["speaker_id"]
        if spk_id not in speaker_mapping:
            speaker_mapping[spk_id] = f"V{len(speaker_mapping) + 1}"

    for segment in merged_segments:
        segment["speaker"] = speaker_mapping[segment["speaker_id"]]

    total_actors = len(speaker_mapping)
    print(f"\n🎉 {total_actors} acteur(s) identifié(s) dans le film !")
    for orig_id, spk_label in speaker_mapping.items():
        print(f"  • Acteur {spk_label} (ID interne: {orig_id})")

    # 8. Extraction des fichiers de référence vocale pour chaque acteur
    print("\nExtraction des échantillons de référence vocale pour chaque acteur...")
    speaker_references = {}

    for speaker_num in range(1, total_actors + 1):
        speaker_name = f"V{speaker_num}"
        reference_audio = AudioSegment.empty()
        accumulated_seconds = 0.0

        for segment in merged_segments:
            if segment["speaker"] != speaker_name:
                continue

            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            chunk = original_audio[start_ms:end_ms]
            chunk_seconds = len(chunk) / 1000.0

            remaining = reference_max_seconds - accumulated_seconds
            if remaining <= 0:
                break

            if chunk_seconds > remaining:
                chunk = chunk[:int(remaining * 1000)]
                chunk_seconds = remaining

            reference_audio += chunk
            reference_audio += AudioSegment.silent(duration=150)
            accumulated_seconds += chunk_seconds

        ref_file = voices_dir / f"{speaker_name}_reference.wav"
        reference_audio.export(str(ref_file), format="wav")
        speaker_references[speaker_name] = str(ref_file)
        print(f"  • Voix de référence {speaker_name} ({accumulated_seconds:.1f}s) -> {ref_file}")

    # 9. Découpage et sauvegarde de tous les segments de dialogue
    print(f"\nDécoupage des {len(merged_segments)} segments de répliques...")
    segments_data = []

    for index, segment in enumerate(merged_segments, start=1):
        speaker = segment["speaker"]
        start = segment["start"]
        end = segment["end"]
        duration = end - start

        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        chunk = original_audio[start_ms:end_ms]

        segment_file = segments_dir / f"segment_{index:05d}_{speaker}.wav"
        chunk.export(str(segment_file), format="wav")

        segments_data.append({
            "index": index,
            "speaker": speaker,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(duration, 3),
            "audio_path": str(segment_file)
        })

    # 10. Sauvegarde des métadonnées
    speakers_info_file = output_dir / "speakers_info.json"
    segments_info_file = output_dir / "segments_info.json"

    with open(speakers_info_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_speakers": total_actors,
            "mapping": speaker_mapping,
            "references": speaker_references
        }, f, ensure_ascii=False, indent=2)

    with open(segments_info_file, "w", encoding="utf-8") as f:
        json.dump(segments_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Diarisation terminée : {len(segments_data)} répliques réparties sur {total_actors} acteur(s).")
    return segments_data, speaker_references


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python step2_diarization.py <chemin_audio> [dossier_sortie]")
        sys.exit(1)

    audio_in = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else "workspace"
    separer_acteurs(audio_in, out_dir)

