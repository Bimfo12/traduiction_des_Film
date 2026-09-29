#!/usr/bin/env python3
"""
Point d'entrée principal pour la traduction et le doublage de film entier.

Exemples d'utilisation :
    python main.py video.mp4
    python main.py video.mp4 --output film_francais.mp4
    python main.py --help
"""

import sys
from pathlib import Path
from pipeline import FilmTranslationPipeline, parse_args


def main():
    # Si aucun argument n'est fourni, inviter l'utilisateur
    if len(sys.argv) == 1:
        print("=" * 70)
        print("🎬 SYSTÈME DE TRADUCTION ET DOUBLAGE VOCAL DE FILMS (VF)")
        print("=" * 70)
        video_input = input("Entrez le chemin du fichier vidéo à traduire : ").strip()
        if not video_input:
            print("❌ Aucun fichier fourni. Arrêt du programme.")
            sys.exit(1)

        video_path = Path(video_input)
        pipeline = FilmTranslationPipeline(video_path=video_path)
        pipeline.run_all()
    else:
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


if __name__ == "__main__":
    main()

