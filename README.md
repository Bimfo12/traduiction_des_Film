# 🎬 Système de Traduction et Doublage Complet de Film en Français (VF)

Ce projet permet de traduire un film entier de l'anglais vers le français avec :
- **Préservation intégrale de la musique de fond et des effets sonores** (grâce à Demucs et au mixage de `main_via_gpt.py`).
- **Identification des acteurs et calage temporel** des dialogues.
- **Synthèse vocale claire et audible** avec Google TTS (`gTTS`) ou clonage vocal (`XTTS v2`).
- **Traduction en lot de n'importe quel dossier de voix** en VF.

---

## 🏗️ Architecture du Projet

```
traduiction_des_Film/
│
├── config.py                              # Configuration globale (seuils, modèles, périphériques)
├── pipeline.py                            # Orchestrateur complet reliant les 5 étapes + musique
├── main.py                                # Point d'entrée simple d'utilisation
├── main_via_gpt.py                        # Script copié depuis english-french-voice
├── traduire_voix.py                       # Traduit tous les fichiers du dossier voices/ en VF
├── requirements.txt                       # Dépendances Python
│
├── modules/
│   ├── step1_extract_audio.py             # Étape 1 : Extraction audio (FFmpeg)
│   ├── audio_separator.py                 # Étape 1b & 4b : Séparation Demucs & mixage musique de fond
│   ├── step2_diarization.py               # Étape 2 : Diarisation & extraction des voix des acteurs
│   ├── step3_translation.py               # Étape 3 : Transcription (Whisper) & Traduction (MarianMT)
│   ├── step4_voice_cloning.py             # Étape 4 : Synthèse vocale gTTS / XTTS & calage temporel
│   └── step5_merge_video.py               # Étape 5 : Fusion vidéo + bande-son mixée finale
│
├── pretrained_models/ -> cache partagé    # Modèle SpeechBrain
├── workspace/                             # Segments, voix extraites, dialogues JSON
└── output/                                # Films finaux doublés en VF
```

---

## 🎵 Préservation de la Musique de Fond (Demucs + Mixage)

Inspiré directement de `main_via_gpt.py` :
1. **Séparation Demucs** : l'audio du film est séparé en deux pistes :
   - `vocals.wav` : voix originale seule (les dialogues en anglais).
   - `no_vocals.wav` : **musique de fond originale, bruitages et ambiances**, sans les voix anglaises.
2. **Traduction & Synthèse VF** : seules les répliques sont traduites et synthétisées en français (`french_voices_only.wav`).
3. **Mixage final (FFmpeg amix)** : la nouvelle voix française est mixée par-dessus la musique de fond originale (`no_vocals.wav`). La musique de fond n'est donc **jamais perdue** !

---

## 🎙️ Traduction de tous les fichiers de voix en VF

Pour traduire automatiquement l'ensemble des fichiers de voix présents dans `workspace/video/voices/` en français (VF) avec la méthode de `main_via_gpt.py` :

```bash
# Avec le script dédié :
python traduire_voix.py

# Ou en spécifiant les dossiers :
python traduire_voix.py --voices-dir workspace/video/voices --output-dir workspace/video/voices_vf

# Ou via main_via_gpt.py directement :
python main_via_gpt.py --voices
```

Ce script produit pour chaque voix :
- Un fichier MP3 et WAV traduit en français (`V1_reference_vf.mp3`, `V1_reference_vf.wav`, etc.).
- Un fichier récapitulatif `recapitulatif_traduction.txt` et `recapitulatif_traduction.json`.

---

## 🚀 Traduire un Film Entier

### 1. Activer l'environnement virtuel
```bash
cd /home/edouard/Bureau/traduiction_des_Film
source venv/bin/activate
```

### 2. Lancer la traduction complète
```bash
python main.py video.mp4
```

### 3. Options avancées du pipeline (`pipeline.py`)

```bash
# Traduction avec préservation de la musique et moteur gTTS (par défaut, recommandé) :
python pipeline.py video.mp4

# Forcer le moteur de clonage vocal XTTS :
python pipeline.py video.mp4 --tts xtts

# Reprendre le pipeline à une étape précise (ex: après avoir ajusté dialogues_traduits.json) :
python pipeline.py video.mp4 --step 4

# Spécifier le nombre connu d'acteurs principaux :
python pipeline.py video.mp4 --speakers 2
```
# traduiction_des_Film
