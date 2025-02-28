import os
import pandas as pd
import re
from typing import List, Tuple

from config import PROC_DIR, RAW_DIR


def filter_artist_sections(artist: str, lyrics: str) -> List[Tuple[str]]:
    '''
    Extracts only the sections of the lyrics (ie verse, chorus, etc) that
    contain the artist's name.

    Parameters
    ----------
    artist: str
        name of the artist to filter sections by
    lyrics: str
        the lyrics of the song to be processed

    Returns
    -------
    matches: List[Tuple[str]]
        a list of tuples of the form (section_name, section_lyrics)
    '''
    pattern = rf"\[([^]]*{re.escape(artist)}[^]]*)\]\s*((?:.(?!\[))*.)"
    matches = re.findall(pattern, lyrics, re.DOTALL | re.IGNORECASE)

    # lyrics don't contain any sections, assign all lyrics to verse
    if not matches:
        matches = [("Verse", lyrics)]

    return matches


def format_section(artist: str, section_header: str, lyrics: str) -> str:
    '''
    Formats the section header, removing any unnecessary information like
    artist name or feature, and returns a single string with the section header
    and lyrics.

    Parameters
    ----------
    artist: str
        name of the artist to filter
    section_header: str
        the header of the section containing the section type and artist name
    lyrics: str
        the lyrics of the sections

    Returns
    -------
    str: a single string of the form "[section_header]\nlyrics"
    '''
    section_mapping = {
        artist.lower(): "Verse",
        "verse": "Verse",
        "rap": "Verse",
        "hook": "Hook",
        "chorus": "Chorus",
        "coro": "Chorus",
        "pre-chorus": "Pre-Chorus",
        "pre-coro": "Pre-Chorus",
        "bridge": "Bridge",
        "outro": "Outro",
        "intro": "Intro",
        "pre-hook": "Pre-Hook",
        "post-chorus": "Post-Chorus",
        "interlude": "Interlude",
        "freestyle": "Freestyle",
        "post commentary": "Post Commentary",
        "skit": "Skit",
        "refrain": "Refrain",
    }

    clean_header = section_header.lower()
    for key, mapped_name in section_mapping.items():
        if key in clean_header:
            return f"[{mapped_name}]\n{lyrics.strip()}"

    # if the section header doesn't match any of the section types, remove the
    # artist name and return the section name
    print("No match found for section header:", section_header)
    clean_header = re.sub(r"\b" + re.escape(artist) + r"\b", "",
                          section_header,
                          flags=re.IGNORECASE).strip()

    return f"[{clean_header}]\n{lyrics.strip()}"


def split_dataset(df: pd.DataFrame, output_dir: str) -> None:
    '''
    Splits the dataset into training and validation sets and saves them to
    disk.

    Parameters
    ----------
    df: pd.DataFrame
        the dataset to split
    output_dir: str
        the directory to save the split datasets to
    '''
    os.makedirs(output_dir, exist_ok=True)

    train_size = int(0.8 * len(df))
    df = df.sample(frac=1, random_state=100).reset_index(drop=True)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    train_df.to_json(os.path.join(output_dir, "train.json"),
                     lines=True, orient="records")
    val_df.to_json(os.path.join(output_dir, "val.json"),
                   lines=True, orient="records")


def process_lyrics(artist: str, lyrics: str) -> str:
    '''
    Processes the lyrics of a song to only include sections that contain the
    artist's name.

    Parameters
    ----------
    artist: str
        name of the artist to filter sections by
    lyrics: str
        the lyrics of the song to be processed

    Returns
    -------
    str: the processed lyrics
    '''
    sections = filter_artist_sections(artist, lyrics)
    formatted_lyrics = [format_section(artist, header, section)
                        for header, section in sections]
    return "\n\n".join(formatted_lyrics)


def preprocess_dataset(artist: str) -> None:
    '''
    Filters the dataset to only include songs by the artist and formats the
    lyrics to only include sections that contain the artist's name.

    Parameters
    ----------
    artist: str
        name of the artist to filter the dataset by

    Returns
    -------
    pd.DataFrame: a pandas dataframe containing the filtered and formatted
    dataset
    '''
    artist_filename = f"{artist.replace(' ', '_').lower()}"
    raw_file_path = os.path.join(RAW_DIR, f"{artist_filename}.csv")
    proc_file_path = os.path.join(PROC_DIR, f"{artist_filename}")

    if os.path.exists(proc_file_path):
        print(f"Processed result already exists for {artist}.")
        return

    print(f"Processing dataset for {artist}...")

    df = pd.read_csv(raw_file_path)
    proc_df = pd.DataFrame({'clean_lyrics': df["lyrics"].apply(
        lambda x: process_lyrics(artist, x))})

    split_dataset(proc_df, proc_file_path)

    print(f"Saved preprocessed dataset for {artist} to {proc_file_path}")
