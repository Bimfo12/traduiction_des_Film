import os
import re
import shutil
import subprocess
from pathlib import Path

from gtts import gTTS
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer

# ============================================================
# CONFIGURATION
# ============================================================

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv(
    "AZURE_SPEECH_REGION",
    "francecentral"
)
AZURE_SPEECH_VOICE = os.getenv(
    "AZURE_SPEECH_VOICE",
    "fr-FR-DeniseNeural"
)

INPUT_FILE = "black-holl.mp3"
WORK_DIR = Path("work")
OUTPUT_FILE = "black-holl-fr.mp3"

# Partie de la voix originale utilisée comme référence.
# ATTENTION : actuellement gTTS n'utilise PAS cette référence
# pour cloner la voix.
REFERENCE_START = 0
REFERENCE_DURATION = 12

# Petit délai entre deux phrases lorsque cela est nécessaire.
# Cela évite que deux phrases commencent exactement au même instant.
MIN_GAP = 0.03


# ============================================================
# OUTILS
# ============================================================

def run_command(command):
    print("\n$", " ".join(map(str, command)))

    subprocess.run(
        command,
        check=True
    )


def get_audio_duration(audio_file):
    """
    Retourne la durée d'un fichier audio en secondes.
    """
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_file),
        ],
        capture_output=True,
        text=True,
        check=True
    )

    try:
        return float(probe.stdout.strip())
    except (ValueError, TypeError):
        return 0.0


# ============================================================
# 1. SEPARER LA VOIX ET LA MUSIQUE
# ============================================================

def separate_audio():
    print("\n==========================================")
    print("Séparation voix / musique avec Demucs")
    print("==========================================")

    WORK_DIR.mkdir(exist_ok=True)

    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Fichier audio introuvable : {input_path}"
        )

    if shutil.which("demucs") is None:
        raise RuntimeError(
            "Demucs n'est pas installé.\n"
            "Installe-le avec : python -m pip install demucs"
        )

    model_name = "htdemucs_ft"

    print(f"\nModèle : {model_name}")
    print("Qualité : haute")
    print("Shifts : 1")
    print("Overlap : 0.15")

    run_command([
    "demucs",

    # Séparation voix / instrumental
    "--two-stems=vocals",

    # Modèle
    "-n",
    model_name,

    # Nombre de shifts
    "--shifts",
    "1",

    # Overlap entre segments
    "--overlap",
    "0.15",

    str(input_path)
])

    base_name = input_path.stem

    vocals = Path(
        "separated",
        model_name,
        base_name,
        "vocals.wav"
    )

    instrumental = Path(
        "separated",
        model_name,
        base_name,
        "no_vocals.wav"
    )

    if not vocals.exists():
        raise FileNotFoundError(
            f"Demucs n'a pas produit la voix : {vocals}"
        )

    if not instrumental.exists():
        raise FileNotFoundError(
            f"Demucs n'a pas produit la musique : {instrumental}"
        )

    print("\n✓ Voix anglaise :")
    print(vocals)

    print("\n✓ Musique sans voix :")
    print(instrumental)

    return vocals, instrumental


# ============================================================
# 2. EXTRAIRE UN EXTRAIT POUR LA REFERENCE
# ============================================================

def create_reference_voice(vocals_file):
    reference_file = WORK_DIR / "reference.wav"

    print("\n==========================================")
    print("Préparation de la voix de référence")
    print("==========================================")

    run_command([
        "ffmpeg",
        "-y",

        "-ss",
        str(REFERENCE_START),

        "-t",
        str(REFERENCE_DURATION),

        "-i",
        str(vocals_file),

        "-ac",
        "1",

        "-ar",
        "24000",

        str(reference_file)
    ])

    return reference_file


# ============================================================
# 3. WHISPER
# ============================================================

def load_whisper():
    print("\n==========================================")
    print("Chargement de Whisper")
    print("==========================================")

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

    return model


# ============================================================
# 4. TRANSCRIPTION
# ============================================================

def transcribe(model, vocals_file):
    print("\n==========================================")
    print("Transcription anglaise par phrase")
    print("==========================================")

    segments, info = model.transcribe(
        str(vocals_file),
        language="en",
        vad_filter=True,
        beam_size=5,
        word_timestamps=True
    )

    results = []

    for segment in segments:

        if not segment.words:
            continue

        current_words = []

        for word in segment.words:

            if not word.word:
                continue

            current_words.append(word)

            text = "".join(
                w.word for w in current_words
            ).strip()

            # ------------------------------------------------
            # Créer un segment uniquement lorsqu'on rencontre
            # un point.
            # ------------------------------------------------
            if text.endswith("."):

                start = current_words[0].start
                end = current_words[-1].end

                item = {
                    "start": float(start),
                    "end": float(end),
                    "text": text
                }

                results.append(item)

                print(
                    f"[{start:.2f}s -> "
                    f"{end:.2f}s] "
                    f"{text}"
                )

                current_words = []

        # ----------------------------------------------------
        # S'il reste des mots sans point à la fin du segment
        # Whisper, on les conserve.
        # ----------------------------------------------------

        if current_words:

            text = "".join(
                w.word for w in current_words
            ).strip()

            if text:

                start = current_words[0].start
                end = current_words[-1].end

                item = {
                    "start": float(start),
                    "end": float(end),
                    "text": text
                }

                results.append(item)

                print(
                    f"[{start:.2f}s -> "
                    f"{end:.2f}s] "
                    f"{text}"
                )

    return results


# ============================================================
# 5. CHARGER LE MODELE DE TRADUCTION
# ============================================================

def load_translation_model():
    print("\n==========================================")
    print("Chargement de MarianMT")
    print("==========================================")

    model_name = "Helsinki-NLP/opus-mt-en-fr"

    tokenizer = MarianTokenizer.from_pretrained(
        model_name
    )

    model = MarianMTModel.from_pretrained(
        model_name
    )

    return tokenizer, model


# ============================================================
# 6. DECOUPER LE TEXTE POUR LA TRADUCTION
# ============================================================

def split_text_for_translation(
    text,
    tokenizer,
    max_tokens=350
):
    text = (text or "").strip()

    if not text:
        return []

    model_limit = getattr(
        tokenizer,
        "model_max_length",
        512
    )

    safe_limit = min(
        max_tokens,
        model_limit - 50
    )

    if safe_limit <= 0:
        safe_limit = model_limit - 50

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []
    current = ""

    def token_count(value):
        return len(
            tokenizer.encode(
                value,
                add_special_tokens=False
            )
        )

    for sentence in sentences:

        if not sentence.strip():
            continue

        candidate = (
            sentence
            if not current
            else f"{current} {sentence}"
        )

        if token_count(candidate) <= safe_limit:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if token_count(sentence) <= safe_limit:
            current = sentence
            continue

        for word in sentence.split():

            candidate = (
                word
                if not current
                else f"{current} {word}"
            )

            if token_count(candidate) <= safe_limit:
                current = candidate

            else:

                if current:
                    chunks.append(current)

                current = word

    if current:
        chunks.append(current)

    return [
        chunk.strip()
        for chunk in chunks
        if chunk and chunk.strip()
    ]


# ============================================================
# 7. TRADUIRE
# ============================================================

def translate_text(
    text,
    tokenizer,
    model
):
    chunks = split_text_for_translation(
        text,
        tokenizer
    )

    if not chunks:
        return ""

    translated_parts = []

    for chunk in chunks:

        inputs = tokenizer(
            [chunk],
            return_tensors="pt",
            padding=True,
            truncation=False
        )

        translated = model.generate(
            **inputs,
            max_length=512,
            num_beams=4
        )

        result = tokenizer.decode(
            translated[0],
            skip_special_tokens=True
        )

        translated_parts.append(result)

    return " ".join(
        part.strip()
        for part in translated_parts
        if part.strip()
    )


def save_transcription(translated_segments):
    """
    Sauvegarde la transcription anglaise et sa traduction française
    dans un fichier texte lisible et modifiable manuellement.

    Les timestamps sont conservés pour garder la synchronisation
    avec la vidéo.
    """
    transcription_file = WORK_DIR / "transcription_en_fr.txt"

    WORK_DIR.mkdir(exist_ok=True)

    with transcription_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write("=" * 70 + "\n")
        f.write("TRANSCRIPTION ANGLAIS -> FRANÇAIS\n")
        f.write("=" * 70 + "\n\n")

        for i, segment in enumerate(
            translated_segments,
            start=1
        ):
            start = segment["start"]
            end = segment["end"]
            english = segment["text"]
            french = segment["french"]

            f.write(
                f"SEGMENT {i}\n"
                f"[{start:.2f}s -> {end:.2f}s]\n"
            )

            f.write(
                f"EN : {english}\n"
            )

            f.write(
                f"FR : {french}\n"
            )

            f.write("\n" + "-" * 70 + "\n\n")

    print(
        "\n✓ Transcription sauvegardée :"
    )
    print(
        transcription_file.resolve()
    )

    return transcription_file


# ============================================================
# 8. CHARGER LE TTS
# ============================================================

def load_voice_model():
    print("\n==========================================")
    print("Chargement de Google Text-to-Speech")
    print("==========================================")

    try:
        from gtts import gTTS

        print("✓ Google TTS chargé.")
        print("✓ Langue : français")
        print("✓ Mode : vitesse naturelle")

        return {
            "type": "gtts",
            "factory": gTTS
        }

    except Exception as exc:
        raise RuntimeError(
            f"Google TTS indisponible : {exc}"
        )


# ============================================================
# 9. GENERER LA VOIX FRANÇAISE
# ============================================================

def generate_voice(
    tts,
    french_text,
    reference_file,
    output_file
):
    """
    Génère la voix française sans modifier sa vitesse.

    IMPORTANT :
    reference_file n'est pas utilisé par gTTS.
    Il sera utile lorsqu'un vrai modèle de voice cloning
    comme XTTS sera utilisé.
    """

    if isinstance(tts, dict) and tts.get("type") == "gtts":

        gtts = tts["factory"]

        print(f"Google TTS : {french_text}")

        speech = gtts(
            text=french_text,
            lang="fr",
            slow=False
        )

        speech.save(
            str(output_file)
        )

        return

    raise TypeError(
        "Moteur Google TTS non compatible"
    )


# ============================================================
# 10. CONSTRUIRE LA PISTE VOCALE
# ============================================================

def build_voice_track(
    translated_segments,
    instrumental_file
):
    print("\n==========================================")
    print("Création de la piste vocale française")
    print("==========================================")

    voice_output = WORK_DIR / "translated_voice.wav"

    # Durée exacte du morceau / de la vidéo
    duration = get_audio_duration(
        instrumental_file
    )

    if duration <= 0:
        raise RuntimeError(
            "Impossible de déterminer la durée du morceau."
        )

    print(
        f"\nDurée de la vidéo : "
        f"{duration:.2f} secondes"
    )

    # --------------------------------------------------------
    # Fichiers audio à mixer
    # --------------------------------------------------------

    inputs = []

    # Silence de la durée complète de la vidéo.
    #
    # Cela garantit que même lorsque toutes les voix
    # sont terminées, la piste garde la bonne durée.
    inputs.extend([
        "-f",
        "lavfi",

        "-t",
        str(duration),

        "-i",
        "anullsrc=r=44100:cl=stereo"
    ])

    filters = []
    audio_inputs = []

    # [0:a] = piste silencieuse de référence
    audio_inputs.append("[0:a]")

    # Position de fin réelle de la dernière voix générée.
    #
    # Elle permet d'éviter que deux phrases se chevauchent
    # lorsque le français est plus long que l'anglais.
    current_voice_end = 0.0

    scheduled_segments = []

    # --------------------------------------------------------
    # Préparer chaque voix
    # --------------------------------------------------------

    for i, segment in enumerate(translated_segments):

        audio_file = WORK_DIR / f"voice_{i}.mp3"

        if not audio_file.exists():
            raise FileNotFoundError(
                f"Voix française absente : {audio_file}"
            )

        natural_duration = get_audio_duration(
            audio_file
        )

        if natural_duration <= 0:
            print(
                f"\n⚠ Segment {i + 1} ignoré : "
                f"durée audio invalide."
            )
            continue

        original_start = float(
            segment["start"]
        )

        original_end = float(
            segment["end"]
        )

        original_slot = (
            original_end - original_start
        )

        # ----------------------------------------------------
        # Position de départ
        # ----------------------------------------------------
        #
        # Cas normal :
        #
        #   original_start > current_voice_end
        #
        # => on garde exactement le timestamp original.
        #
        # Cas où le français est plus long :
        #
        #   original_start < current_voice_end
        #
        # => on attend que la phrase précédente soit terminée.
        #
        start = max(
            original_start,
            current_voice_end + MIN_GAP
        )

        # Ne pas démarrer après la fin de la vidéo.
        if start >= duration:
            print(
                f"\n⚠ Segment {i + 1} ignoré : "
                f"il commence après la fin de la vidéo."
            )
            continue

        natural_end = (
            start + natural_duration
        )

        # ----------------------------------------------------
        # Informations
        # ----------------------------------------------------

        print(
            f"\nSegment {i + 1}"
        )

        print(
            f"  EN : {segment['text']}"
        )

        print(
            f"  FR : {segment['french']}"
        )

        print(
            f"  Timestamp anglais : "
            f"{original_start:.2f}s -> "
            f"{original_end:.2f}s"
        )

        print(
            f"  Durée disponible : "
            f"{original_slot:.2f}s"
        )

        print(
            f"  Durée naturelle FR : "
            f"{natural_duration:.2f}s"
        )

        if natural_duration < original_slot:

            silence_after = (
                original_slot -
                natural_duration
            )

            print(
                f"  ✓ La voix finit naturellement."
            )

            print(
                f"  ✓ Silence après la phrase : "
                f"{silence_after:.2f}s"
            )

        elif natural_duration > original_slot:

            overflow = (
                natural_duration -
                original_slot
            )

            print(
                f"  ⚠ La phrase française est "
                f"{overflow:.2f}s plus longue."
            )

            print(
                f"  ✓ Vitesse NON modifiée."
            )

            print(
                f"  ✓ Le prochain dialogue sera "
                f"décalé pour éviter le chevauchement."
            )

        else:

            print(
                f"  ✓ Durée parfaite."
            )

        scheduled_segments.append({
            "index": i,
            "file": audio_file,
            "start": start,
            "duration": natural_duration,
            "end": natural_end
        })

        # La prochaine phrase ne pourra pas commencer
        # avant la fin réelle de celle-ci.
        current_voice_end = natural_end

    # --------------------------------------------------------
    # Créer les délais
    # --------------------------------------------------------

    for scheduled in scheduled_segments:

        index = scheduled["index"]
        start = scheduled["start"]

        delay = max(
            0,
            int(round(start * 1000))
        )

        input_index = index + 1

        filters.append(
            f"[{input_index}:a]"
            f"adelay={delay}:all=1"
            f"[v{index}]"
        )

        audio_inputs.append(
            f"[v{index}]"
        )

        inputs.extend([
            "-i",
            str(scheduled["file"])
        ])

    # --------------------------------------------------------
    # Vérifier qu'il existe des voix
    # --------------------------------------------------------

    if not scheduled_segments:
        raise RuntimeError(
            "Aucune voix française à mixer."
        )

    # --------------------------------------------------------
    # Mélanger toutes les voix
    # --------------------------------------------------------

    filters.append(
        "".join(audio_inputs)
        +
        f"amix=inputs={len(audio_inputs)}:"
        "duration=longest:"
        "dropout_transition=0:"
        "normalize=0"
        "[mixed]"
    )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # On limite la piste vocale à la durée de la vidéo.
    #
    # Cela permet de conserver exactement la longueur
    # originale sans accélérer les voix.
    # --------------------------------------------------------

    filters.append(
        "[mixed]"
        f"atrim=duration={duration:.6f},"
        "asetpts=PTS-STARTPTS"
        "[out]"
    )

    filter_complex = ";".join(
        filters
    )

    command = [
        "ffmpeg",
        "-y"
    ]

    command.extend(inputs)

    command.extend([
        "-filter_complex",
        filter_complex,

        "-map",
        "[out]",

        "-c:a",
        "pcm_s16le",

        "-t",
        str(duration),

        str(voice_output)
    ])

    run_command(command)

    print(
        f"\n✓ Piste vocale créée : "
        f"{voice_output}"
    )

    print(
        f"✓ Durée finale de la piste vocale : "
        f"{duration:.2f}s"
    )

    return voice_output


# ============================================================
# 11. MELANGER MUSIQUE + VOIX
# ============================================================

def mix_audio(
    instrumental_file,
    voice_file
):
    print("\n==========================================")
    print("Mixage final")
    print("==========================================")

    # Ne JAMAIS utiliser le fichier original ici.
    if (
        Path(instrumental_file).resolve()
        ==
        Path(INPUT_FILE).resolve()
    ):
        raise RuntimeError(
            "ERREUR : la piste musicale correspond encore "
            "au fichier original. La voix anglaise serait conservée."
        )

    if not Path(instrumental_file).exists():
        raise FileNotFoundError(
            f"Musique introuvable : {instrumental_file}"
        )

    if not Path(voice_file).exists():
        raise FileNotFoundError(
            f"Voix française introuvable : {voice_file}"
        )

    duration = get_audio_duration(
        instrumental_file
    )

    run_command([
        "ffmpeg",
        "-y",

        # MUSIQUE SANS VOIX
        "-i",
        str(instrumental_file),

        # NOUVELLE VOIX FRANÇAISE
        "-i",
        str(voice_file),

        "-filter_complex",
        (
            "[0:a]"
            "volume=1.0"
            "[music];"

            "[1:a]"
            "volume=1.0"
            "[voice];"

            "[music][voice]"
            "amix="
            "inputs=2:"
            "duration=longest:"
            "dropout_transition=0:"
            "normalize=0"
            "[a];"

            "[a]"
            f"atrim=duration={duration:.6f},"
            "asetpts=PTS-STARTPTS"
            "[out]"
        ),

        "-map",
        "[out]",

        "-c:a",
        "libmp3lame",

        "-b:a",
        "192k",

        # Durée finale = durée originale
        "-t",
        str(duration),

        str(OUTPUT_FILE)
    ])

    print("\n==========================================")
    print("✓ FICHIER FINAL")
    print("==========================================")

    print(
        Path(OUTPUT_FILE).resolve()
    )


# ============================================================
# 12. PROGRAMME PRINCIPAL
# ============================================================

def main():
    import sys
    if len(sys.argv) > 1 and any(arg.startswith("--voices") for arg in sys.argv):
        from traduire_voix import traduire_dossier_voix
        v_dir = "/home/edouard/Bureau/traduiction_des_Film/workspace/video/voices"
        for i, a in enumerate(sys.argv):
            if a in ["--voices", "--voices-dir"] and i + 1 < len(sys.argv) and not sys.argv[i+1].startswith("-"):
                v_dir = sys.argv[i+1]
        traduire_dossier_voix(voices_dir=v_dir)
        return

    print("\n==========================================")
    print(" TRADUCTEUR AUDIO ANGLAIS -> FRANÇAIS")
    print("==========================================")

    if not os.path.exists(INPUT_FILE):
        default_voices = Path("/home/edouard/Bureau/traduiction_des_Film/workspace/video/voices")
        if default_voices.exists():
            print(f"Fichier {INPUT_FILE} introuvable, bascule automatique sur la traduction des voix dans : {default_voices}")
            from traduire_voix import traduire_dossier_voix
            traduire_dossier_voix(voices_dir=default_voices)
            return
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Séparer voix et musique
    # --------------------------------------------------------

    vocals_file, instrumental_file = (
        separate_audio()
    )

    # --------------------------------------------------------
    # Créer une référence
    # --------------------------------------------------------

    reference_file = (
        create_reference_voice(
            vocals_file
        )
    )

    # --------------------------------------------------------
    # Whisper
    # --------------------------------------------------------

    whisper = load_whisper()

    segments = transcribe(
        whisper,
        vocals_file
    )

    if not segments:
        print("\nAucune parole détectée.")
        return

    # --------------------------------------------------------
    # Traduction
    # --------------------------------------------------------

    tokenizer, translation_model = (
        load_translation_model()
    )

    translated_segments = []

    print("\n==========================================")
    print("Traduction")
    print("==========================================")

    for i, segment in enumerate(segments):

        english = segment["text"]

        french = translate_text(
            english,
            tokenizer,
            translation_model
        )

        print()
        print(
            "EN :",
            english
        )

        print(
            "FR :",
            french
        )

        translated_segments.append({
            **segment,
            "french": french
        })

    # --------------------------------------------------------
    # Sauvegarder anglais + français
    # --------------------------------------------------------

    save_transcription(
        translated_segments
    )

    # --------------------------------------------------------
    # Génération de la voix
    # --------------------------------------------------------

    tts = load_voice_model()

    print("\n==========================================")
    print("Génération de la voix française")
    print("==========================================")

    for i, segment in enumerate(
        translated_segments
    ):

        output_file = (
            WORK_DIR /
            f"voice_{i}.mp3"
        )

        print(
            f"\nSegment {i + 1} : "
            f"{segment['french']}"
        )

        generate_voice(
            tts,
            segment["french"],
            reference_file,
            output_file
        )

    # --------------------------------------------------------
    # Construction de la piste vocale
    # --------------------------------------------------------

    voice_file = build_voice_track(
        translated_segments,
        instrumental_file
    )

    # --------------------------------------------------------
    # Mixage final
    # --------------------------------------------------------

    mix_audio(
        instrumental_file,
        voice_file
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()