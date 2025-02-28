import argparse
import os

from config import RAW_DIR, PROC_DIR, TOK_DIR
from pipeline import (
    get_artist_dataset,
    preprocess_dataset,
    tokenize_dataset,
    fine_tune_model
)


def main(artist: str) -> None:
    print(f"Fine-tuning GPT2 model on lyrics by {artist}...")
    artist_filename = f"{artist.replace(' ', '_').lower()}"
    raw_file_path = os.path.join(RAW_DIR, f"{artist_filename}.csv")
    proc_file_path = os.path.join(PROC_DIR, f"{artist_filename}")
    token_file_path = os.path.join(TOK_DIR, f"{artist_filename}")

    if not os.path.exists(raw_file_path):
        get_artist_dataset(artist)

    if not os.path.exists(proc_file_path):
        preprocess_dataset(artist)

    if not os.path.exists(token_file_path):
        tokenize_dataset(artist)

    fine_tune_model(artist)
    print("Model fine-tuning complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT2 model on an artist's lyrics.")
    parser.add_argument("-a", "--artist", type=str, required=True,
                        help="The name of the artist to fine-tune model")
    args = parser.parse_args()

    main(args.artist)
