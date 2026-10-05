"""
Étape 2 : Diarisation des locuteurs et extraction des références vocales des acteurs.
Basé sur la logique de diarisation de traductionFilm/dubbing.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Permet l'import direct de config quel que soit le dossier d'exécution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torchaudio
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier

import config

# Patch de sécurité pour torchaudio.load afin d'éviter l'erreur libnvrtc.so.13 / torchcodec
_original_torchaudio_load = torchaudio.load


def _safe_torchaudio_load(filepath, *args, **kwargs):
    """
    Patch pour forcer la lecture audio via soundfile afin d'éviter
    les erreurs torchcodec / libnvrtc sur certaines installations.
    """
    # Si ce n'est pas un chemin, on laisse faire l'implémentation originale
    if not isinstance(filepath, (str, Path, os.PathLike)):
        return _original_torchaudio_load(filepath, *args, **kwargs)

    try:
        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data).T.contiguous()
        return waveform, int(sr)
    except Exception as first_error:
        # Tentative de repli avec le backend soundfile si disponible
        try:
            kwargs["backend"] = "soundfile"
            return _original_torchaudio_load(filepath, *args, **kwargs)
        except Exception:
            raise RuntimeError(
                f"Impossible de charger l'audio : {filepath}"
            ) from first_error


torchaudio.load = _safe_torchaudio_load


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance cosine entre deux vecteurs."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(1.0 - float(np.dot(a, b) / denom))


def _merge_close_speaker_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict]:
    """
    Fusionne les clusters dont les centroïdes sont à une distance cosine
    inférieure à *threshold*. Retourne les nouveaux labels et un mapping.
    """
    unique_labels = np.unique(labels)
    centroids = {
        int(lbl): X[labels == lbl].mean(axis=0)
        for lbl in unique_labels
    }

    parent: dict[int, int] = {int(lbl): int(lbl) for lbl in unique_labels}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    sorted_labels = sorted(int(l) for l in unique_labels)
    for i, la in enumerate(sorted_labels):
        for lb in sorted_labels[i + 1:]:
            if _cosine_distance(centroids[la], centroids[lb]) < threshold:
                union(la, lb)

    root_to_new: dict[int, int] = {}
    new_labels = np.empty_like(labels)
    for lbl in unique_labels:
        root = find(int(lbl))
        if root not in root_to_new:
            root_to_new[root] = len(root_to_new)
        new_labels[labels == lbl] = root_to_new[root]

    mapping = {int(old): root_to_new[find(int(old))] for old in unique_labels}
    return new_labels, mapping


def _read_wav_slice(
    audio_path: Path,
    info,
    start_sec: float,
    end_sec: float
) -> np.ndarray:
    """
    Lit une portion d'un fichier audio sans charger tout le fichier.
    Retourne un tableau numpy de shape (frames, channels).
    """
    sr = info.samplerate
    start_frame = int(max(0, round(start_sec * sr)))
    end_frame = int(min(info.frames, round(end_sec * sr)))

    if end_frame <= start_frame:
        return np.zeros((0, max(1, info.channels)), dtype="float32")

    data, _ = sf.read(
        str(audio_path),
        start=start_frame,
        frames=end_frame - start_frame,
        dtype="float32",
        always_2d=True,
    )
    return data


def _write_wav(path: Path, data: np.ndarray, sample_rate: int) -> None:
    """
    Écrit un tableau numpy dans un fichier WAV PCM 16-bit.
    """
    if data.size == 0:
        data = np.zeros((1, 1), dtype="float32")

    # Sécurite anti-clipping
    data = np.clip(data, -1.0, 1.0)

    # Soundfile préfère parfois 1D pour le mono
    if data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]

    sf.write(str(path), data, sample_rate, subtype="PCM_16")


def _workspace_from_vocals(p: Path) -> Path:
    """
    Déduit le dossier de travail depuis un chemin vocals.wav.

    Gère par exemple :
        workspace/video/htdemucs/audio_original/vocals.wav
        -> workspace/video
    """
    if "htdemucs" in p.parts:
        try:
            return p.parents[2]
        except IndexError:
            return p.parent
    return p.parent


def _resolve_device(device: str) -> str:
    """
    Résout le périphérique, avec fallback CPU si CUDA n'est pas disponible.
    """
    if device and device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️  CUDA demandé mais indisponible. Bascule automatique sur CPU.")
        return "cpu"
    return device or "cpu"


def _normalize_rms(data: np.ndarray, target_rms: float = 0.12) -> np.ndarray:
    """
    Normalise le niveau RMS d'un signal audio.
    Utile pour avoir des références vocales plus stables.
    """
    if data.size == 0:
        return data

    data = data.astype(np.float32)
    rms = float(np.sqrt(np.mean(data ** 2)))

    if rms < 1e-6:
        return data

    gain = target_rms / rms

    # Limite le gain pour ne pas amplifier excessivement le bruit.
    gain = min(max(gain, 0.25), 8.0)

    data = data * gain
    return np.clip(data, -0.95, 0.95).astype(np.float32)


def _enforce_no_overlap(
    segments: List[Dict],
    guard_sec: float = 0.05
) -> List[Dict]:
    """
    Garantit que les segments ne se chevauchent pas.
    Si deux segments de locuteurs différents se chevauchent,
    on insère une petite garde de silence.
    """
    if not segments:
        return []

    segments = sorted(segments, key=lambda s: (s["start"], s["end"]))
    out: List[Dict] = []

    for seg in segments:
        seg = seg.copy()

        while out:
            prev = out[-1]

            # Pas de chevauchement
            if seg["start"] >= prev["end"] + guard_sec:
                break

            # Même locuteur : fusion
            if prev.get("speaker_id") == seg.get("speaker_id"):
                prev["end"] = max(prev["end"], seg["end"])
                seg = None
                break

            # Locuteurs différents : on réduit le segment précédent
            allowed_end = seg["start"] - guard_sec

            if allowed_end > prev["start"] + 0.10:
                prev["end"] = allowed_end
                break

            # Sinon le segment précédent est trop court, on le supprime
            out.pop()

        if seg is None:
            continue

        if out:
            seg["start"] = max(seg["start"], out[-1]["end"] + guard_sec)

        if seg["end"] > seg["start"] + 0.10:
            out.append(seg)

    return out


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance cosine entre deux vecteurs."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(1.0 - float(np.dot(a, b) / denom))


def _merge_close_speaker_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict]:
    """
    Fusionne les clusters dont les centroïdes sont à une distance cosine
    inférieure à *threshold*. Retourne les nouveaux labels et un mapping.
    """
    unique_labels = np.unique(labels)
    centroids = {
        int(lbl): X[labels == lbl].mean(axis=0)
        for lbl in unique_labels
    }

    parent: dict[int, int] = {int(lbl): int(lbl) for lbl in unique_labels}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    sorted_labels = sorted(int(l) for l in unique_labels)
    for i, la in enumerate(sorted_labels):
        for lb in sorted_labels[i + 1:]:
            if _cosine_distance(centroids[la], centroids[lb]) < threshold:
                union(la, lb)

    root_to_new: dict[int, int] = {}
    new_labels = np.empty_like(labels)
    for lbl in unique_labels:
        root = find(int(lbl))
        if root not in root_to_new:
            root_to_new[root] = len(root_to_new)
        new_labels[labels == lbl] = root_to_new[root]

    mapping = {int(old): root_to_new[find(int(old))] for old in unique_labels}
    return new_labels, mapping


def _choose_best_reference(
    speaker_name: str,
    merged_segments: List[Dict],
    audio_path: Path,
    audio_info,
    max_seconds: float,
    min_seconds: float = 3.0,
    target_seconds: float = 7.0,
) -> tuple[np.ndarray | None, float]:
    """
    Choisit le meilleur segment continu pour servir de référence vocale.

    Objectif :
    - éviter de mélanger plusieurs émotions / bruits / qualités ;
    - obtenir une voix plus uniforme pour le même acteur ;
    - fournir une référence propre au clonage XTTS.

    Stratégie :
    1. Cherche d'abord parmi les segments >= min_seconds (idéal)
    2. Fallback progressif vers >= 1.5 s puis >= 1.0 s si aucun segment long
    """
    sr = audio_info.samplerate

    def _collect_candidates(floor_sec: float) -> list:
        result = []
        for segment in merged_segments:
            if segment.get("speaker") != speaker_name:
                continue

            duration = segment["end"] - segment["start"]
            if duration < floor_sec:
                continue

            data = _read_wav_slice(
                audio_path,
                audio_info,
                segment["start"],
                segment["end"],
            )

            if data.shape[0] == 0:
                continue

            mono = data.mean(axis=1).astype(np.float32)
            rms = float(np.sqrt(np.mean(mono * mono)))

            # Rejette les segments trop faibles ou saturés
            if rms < 0.004 or rms > 0.85:
                continue

            # Estime la stabilité énergétique du segment
            frame = max(1, int(0.05 * sr))

            if mono.size >= frame * 4:
                usable = mono[: (mono.size // frame) * frame]
                frames_arr = usable.reshape(-1, frame)
                frame_rms = np.sqrt(np.mean(frames_arr * frames_arr, axis=1) + 1e-12)
                instability = float(np.std(frame_rms) / (np.mean(frame_rms) + 1e-12))
            else:
                instability = 1.0

            score = (
                -abs(duration - target_seconds)
                - 3.0 * instability
                + min(rms, 0.25) * 4.0
            )
            result.append((score, mono, duration))
        return result

    # Essai 1 : seuil idéal
    candidates = _collect_candidates(min_seconds)
    # Fallback 1 : seuil réduit à 1.5 s
    if not candidates:
        candidates = _collect_candidates(1.5)
    # Fallback 2 : seuil minimal à 1.0 s
    if not candidates:
        candidates = _collect_candidates(1.0)

    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda x: x[0], reverse=True)

    _, mono, duration = candidates[0]

    # Limite la durée de la référence
    max_frames = int(max_seconds * sr)
    if mono.size > max_frames:
        start = (mono.size - max_frames) // 2
        mono = mono[start : start + max_frames]

    # Normalisation RMS pour homogénéiser le niveau
    mono = _normalize_rms(mono.reshape(-1, 1), target_rms=0.12)

    return mono, float(mono.shape[0] / sr)

def trouver_audio_voix(
    audio_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    require_vocals: bool = True,
) -> Tuple[Path, Path]:
    """
    Identifie et résout la piste audio vocale séparée de la musique (vocals.wav).
    Recherche en priorité :
    1. Si audio_path est fourni et pointe directement vers un vocals.wav existant.
    2. Les chemins standards Demucs :
       - /home/edouard/Bureau/traduiction_des_Film/workspace/video/htdemucs/audio_original/vocals.wav
       - /home/edouard/Bureau/traduiction_des_Film/workspace/video2/htdemucs/audio_original/vocals.wav
    3. Les pistes dans output_dir ou audio_path.parent (htdemucs/.../vocals.wav ou vocals.wav).

    Retourne:
        Tuple (chemin_vocals, output_dir_resolu)
    """
    known_candidates = [
        config.DEFAULT_WORK_DIR / "video" / "htdemucs" / "audio_original" / "vocals.wav",
        config.DEFAULT_WORK_DIR / "video2" / "htdemucs" / "audio_original" / "vocals.wav",
    ]

    resolved_audio: Path | None = None
    resolved_output: Path | None = None

    if output_dir is not None:
        resolved_output = Path(output_dir).resolve()

    if audio_path is not None:
        audio_str = str(audio_path).strip()
        # Si c'est un identifiant court comme 'video' ou 'video2'
        if audio_str in ["video", "video2"]:
            candidate = config.DEFAULT_WORK_DIR / audio_str / "htdemucs" / "audio_original" / "vocals.wav"
            if not candidate.is_file():
                candidate = config.DEFAULT_WORK_DIR / audio_str / "vocals.wav"
            if candidate.is_file():
                resolved_audio = candidate
                if resolved_output is None:
                    resolved_output = config.DEFAULT_WORK_DIR / audio_str
        else:
            p = Path(audio_str).resolve()
            if p.is_file():
                # Si c'est déjà un fichier vocals.wav
                if p.name == "vocals.wav":
                    resolved_audio = p
                    if resolved_output is None:
                        # Si dans un dossier htdemucs/audio_original/vocals.wav
                        if "htdemucs" in p.parts and len(p.parents) >= 3:
                            resolved_output = p.parents[2]
                        else:
                            resolved_output = p.parent
                else:
                    # C'est un audio original (avec musique) : vérifier si vocals.wav existe à côté
                    search_locations = [
                        p.parent / "htdemucs" / "audio_original" / "vocals.wav",
                        p.parent / "htdemucs" / p.stem / "vocals.wav",
                        p.parent / "vocals.wav",
                    ]
                    for loc in search_locations:
                        if loc.is_file():
                            print(f"✨ Piste vocale séparée détectée automatiquement : {loc}")
                            resolved_audio = loc
                            break
                    if resolved_audio is None:
                        resolved_audio = p
                    if resolved_output is None:
                        resolved_output = p.parent
            elif p.is_dir():
                # Si un dossier de travail a été passé en paramètre
                search_locations = [
                    p / "htdemucs" / "audio_original" / "vocals.wav",
                    p / "vocals.wav",
                ]
                for loc in search_locations:
                    if loc.is_file():
                        resolved_audio = loc
                        break
                if resolved_output is None:
                    resolved_output = p

    # Si output_dir est défini mais audio_path n'a pas encore trouvé de vocals.wav
    if resolved_output is not None and (resolved_audio is None or resolved_audio.name != "vocals.wav"):
        search_locations = [
            resolved_output / "htdemucs" / "audio_original" / "vocals.wav",
            resolved_output / "vocals.wav"
        ]
        for loc in search_locations:
            if loc.is_file():
                print(f"✨ Piste vocale séparée trouvée dans le dossier de travail : {loc}")
                resolved_audio = loc
                break

    # Si toujours non résolu, vérifier les chemins connus par défaut
    if resolved_audio is None:
        for cand in known_candidates:
            if cand.is_file():
                resolved_audio = cand
                if resolved_output is None:
                    resolved_output = _workspace_from_vocals(cand)
                print(f"✨ Utilisation de la piste vocale séparée : {resolved_audio}")
                break

    if resolved_audio is None or not resolved_audio.is_file():
        raise FileNotFoundError(
            f"Impossible de trouver la piste vocale séparée de la musique.\n"
            f"Veuillez vous assurer que l'un de ces fichiers existe :\n"
            f"  - {known_candidates[0]}\n"
            f"  - {known_candidates[1]}\n"
        )

    if resolved_output is None:
        resolved_output = _workspace_from_vocals(resolved_audio)

    # Vérification de require_vocals
    if resolved_audio.stem.lower() != "vocals":
        if require_vocals:
            raise FileNotFoundError(
                f"Aucune piste vocale séparée valide trouvée.\n"
                f"Fichier reçu : {resolved_audio}\n\n"
                f"Il faut un fichier type vocals.wav ou vocals.flac.\n"
                f"Génère-le avec Demucs si nécessaire."
            )
        else:
            print(
                "⚠️ Aucune piste vocale séparée trouvée. "
                "Utilisation de l'audio fourni.\n"
                "La diarisation risque d'être dégradée à cause de la musique."
            )

    return resolved_audio, resolved_output


def separer_acteurs(
    audio_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str = config.DEVICE,
    energy_threshold: float = config.ENERGY_THRESHOLD,
    cluster_distance_threshold: float = config.CLUSTER_DISTANCE_THRESHOLD,
    number_of_speakers: int | None = config.NUMBER_OF_SPEAKERS,
    reference_max_seconds: float = config.REFERENCE_MAX_SECONDS,
    merge_gap_seconds: float = config.MERGE_GAP_SECONDS,
    min_segment_seconds: float = config.MIN_SEGMENT_SECONDS,
) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Détecte les acteurs dans la piste vocale isolée (séparée de la musique),
    découpe les segments parlés, et extrait pour chaque acteur un échantillon
    vocal de référence pour le clonage.

    Args:
        audio_path: Chemin du fichier audio (idéalement vocals.wav séparé par Demucs).
        output_dir: Dossier de travail pour enregistrer les segments et références.

    Returns:
        Tuple (segments_data, speaker_references):
            - segments_data: Liste de dictionnaires décrivant chaque réplique
            - speaker_references: Dictionnaire { 'V1': '/chemin/vers/V1_reference.wav', ... }
    """
    audio_path, output_dir = trouver_audio_voix(audio_path, output_dir, require_vocals=True)
    device = _resolve_device(device)

    segments_dir = output_dir / "segments"
    voices_dir = output_dir / "voices"
    segments_dir.mkdir(parents=True, exist_ok=True)
    voices_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("👥 ÉTAPE 2 : SÉPARATION DES ACTEURS & EXTRACTION DES VOIX")
    print("=" * 60)
    print(f"Piste vocale traitée : {audio_path}")
    print(f"Dossier de travail   : {output_dir}")
    print(f"Périphérique         : {device}")

    # 1. Lecture des informations audio sans charger tout le fichier via pydub
    print("Lecture des informations audio...")
    audio_info = sf.info(str(audio_path))
    audio_duration_sec = audio_info.frames / audio_info.samplerate
    print(
        f"Durée totale : {audio_duration_sec:.2f} secondes "
        f"({audio_duration_sec / 60:.2f} minutes)"
    )
    print(f"Échantillonnage : {audio_info.samplerate} Hz")
    print(f"Canaux : {audio_info.channels}")

    # 2. Préparation du modèle SpeechBrain
    print("Chargement du modèle d'empreinte vocale SpeechBrain...")
    pretrained_cache = config.BASE_DIR / "pretrained_models" / "spkrec-ecapa-voxceleb"
    pretrained_cache.parent.mkdir(parents=True, exist_ok=True)
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
    
    # Important :
    # si step_size < window_size, les fenêtres se chevauchent.
    # Cela peut créer des segments de diarisation qui se chevauchent.
    # Pour une version stable, on utilise step_size = window_size.
    step_size = window_size

    # 4. Recherche des fenêtres d'énergie vocale
    print("Recherche des fenêtres de dialogue...")

    min_tail_size = int(getattr(config, "MIN_TAIL_SECONDS", 0.25) * sample_rate)

    speech_windows = []
    num_samples = waveform.numel()

    if num_samples == 0:
        raise RuntimeError("L'audio chargé est vide.")

    if num_samples < window_size:
        # Cas d'un audio très court : on teste le fichier entier
        energy = torch.sqrt(torch.mean(waveform ** 2)).item()
        if energy > energy_threshold:
            speech_windows.append({
                "start_sample": 0,
                "end_sample": num_samples,
                "start": 0.0,
                "end": num_samples / sample_rate,
            })
    else:
        last_start = num_samples - window_size

        for start_sample in range(0, last_start + 1, step_size):
            end_sample = start_sample + window_size
            chunk = waveform[start_sample:end_sample]
            energy = torch.sqrt(torch.mean(chunk ** 2)).item()

            if energy > energy_threshold:
                speech_windows.append({
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "start": start_sample / sample_rate,
                    "end": end_sample / sample_rate,
                })

        # Gestion de la queue finale si elle est suffisante
        processed_until = ((last_start // step_size) * step_size) + window_size
        if processed_until < num_samples:
            tail = waveform[processed_until:num_samples]
            if tail.numel() >= min_tail_size:
                energy = torch.sqrt(torch.mean(tail ** 2)).item()
                if energy > energy_threshold:
                    speech_windows.append({
                        "start_sample": processed_until,
                        "end_sample": num_samples,
                        "start": processed_until / sample_rate,
                        "end": num_samples / sample_rate,
                    })

    print(f"Fenêtres de parole détectées : {len(speech_windows)}")

    if not speech_windows:
        raise RuntimeError(
            "Aucun passage parlé détecté. Vérifie le volume audio ou abaisse ENERGY_THRESHOLD."
        )

    # 5. Extraction des embeddings vocaux
    print("Extraction des empreintes vocales des acteurs...")

    embeddings = []
    total_windows = len(speech_windows)

    for idx, item in enumerate(speech_windows, start=1):
        if idx % 20 == 0 or idx == total_windows:
            print(
                f"\rTraitement empreintes : {idx}/{total_windows} "
                f"({idx * 100 // total_windows}%)",
                end="",
                flush=True,
            )

        chunk = waveform[item["start_sample"]:item["end_sample"]]
        audio_batch = chunk.unsqueeze(0).to(device)

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

    # Fusion des clusters trop proches (stabilise les locuteurs)
    merge_threshold = float(getattr(__import__("config"), "SPEAKER_MERGE_DISTANCE_THRESHOLD", 0.0))
    if merge_threshold > 0 and len(np.unique(labels)) > 1:
        print(f"Fusion des clusters trop proches (seuil cosine < {merge_threshold:.2f})...")
        labels, _ = _merge_close_speaker_clusters(X, labels, merge_threshold)
        print(f"Locuteurs après fusion : {len(np.unique(labels))}")

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

    if not merged_segments:
        raise RuntimeError(
            "Aucun segment de dialogue valide après filtrage.\n"
            "Essaie de diminuer MIN_SEGMENT_SECONDS ou ENERGY_THRESHOLD."
        )

    # Supprime tout chevauchement résiduel entre segments
    guard_sec = float(getattr(config, "SEGMENT_GUARD_SECONDS", 0.05))
    merged_segments = _enforce_no_overlap(merged_segments, guard_sec=guard_sec)

    # Nouveau filtrage après nettoyage
    merged_segments = [
        s for s in merged_segments
        if (s["end"] - s["start"]) >= min_segment_seconds
    ]

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

    ref_min_seconds = float(getattr(config, "REFERENCE_MIN_SECONDS", 3.0))
    ref_target_seconds = float(getattr(config, "REFERENCE_TARGET_SECONDS", 7.0))

    for speaker_num in range(1, total_actors + 1):
        speaker_name = f"V{speaker_num}"

        ref_data, ref_seconds = _choose_best_reference(
            speaker_name=speaker_name,
            merged_segments=merged_segments,
            audio_path=audio_path,
            audio_info=audio_info,
            max_seconds=reference_max_seconds,
            min_seconds=ref_min_seconds,
            target_seconds=ref_target_seconds,
        )

        # Fallback : si aucun segment propre assez long, prendre le premier segment utile
        if ref_data is None:
            for segment in merged_segments:
                if segment["speaker"] != speaker_name:
                    continue

                duration = segment["end"] - segment["start"]
                if duration < 1.0:
                    continue

                data = _read_wav_slice(
                    audio_path,
                    audio_info,
                    segment["start"],
                    segment["end"],
                )

                if data.shape[0] == 0:
                    continue

                mono = data.mean(axis=1).astype(np.float32)
                ref_data = _normalize_rms(mono.reshape(-1, 1), target_rms=0.12)
                ref_seconds = float(ref_data.shape[0] / audio_info.samplerate)
                break

        # Dernier fallback : silence d'une seconde
        if ref_data is None:
            ref_data = np.zeros(
                (int(1.0 * audio_info.samplerate), 1),
                dtype=np.float32,
            )
            ref_seconds = 1.0

        ref_file = voices_dir / f"{speaker_name}_reference.wav"
        _write_wav(ref_file, ref_data, audio_info.samplerate)

        speaker_references[speaker_name] = str(ref_file.resolve())

        print(
            f"  • Voix de référence {speaker_name} "
            f"({ref_seconds:.1f}s) -> {ref_file}"
        )

    # 9. Découpage et sauvegarde de tous les segments de dialogue
    print(f"\nDécoupage des {len(merged_segments)} segments de répliques...")

    segments_data = []

    for index, segment in enumerate(merged_segments, start=1):
        speaker = segment["speaker"]
        start = segment["start"]
        end = segment["end"]
        duration = end - start

        segment_file = segments_dir / f"segment_{index:05d}_{speaker}.wav"

        data = _read_wav_slice(audio_path, audio_info, start, end)
        _write_wav(segment_file, data, audio_info.samplerate)

        segments_data.append({
            "index": index,
            "speaker": speaker,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(duration, 3),
            "audio_path": str(segment_file.resolve()),
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

