import os
import pandas as pd

from config import RAW_DIR
from datasets import load_dataset


def get_artist_dataset(artist: str) -> None:
    '''
    Gets the 'genuis' dataset from huggingface and filters it to only include
    songs by specified artist. If the dataset has already been downloaded it
    skips the download and uses the existing file in later steps.

    Parameters
    ----------
    artist: str
        The name of the artist to filter the dataset by.
    '''
    artist_filename = f"{artist.replace(' ', '_').lower()}.csv"
    raw_file_path = os.path.join(RAW_DIR, artist_filename)

    if os.path.exists(raw_file_path):
        print(f"File already exists for {artist}. Loading from disk...")
        return

    print(f"Fetching dataset for {artist}...")
    genius_ds = load_dataset("sebastiandizon/genius-song-lyrics",
                             split="train", streaming=True)

    filtered_data = [song for song in genius_ds
                     if artist.lower() in song.get('artist', "")
                     or artist.lower() in song.get('features', "")
                     ]

    if not filtered_data:
        raise ValueError(f"No songs by '{artist}' found in the dataset.")

    df = pd.DataFrame(filtered_data)
    df.to_csv(raw_file_path, index=False)
    print(f"Saved dataset for {artist} to {raw_file_path}")
