from transformers import GPT2Tokenizer
from datasets import load_dataset, Features, Sequence, Value
import os
from typing import Dict

from config import PROC_DIR, TOK_DIR


def build_split_dict(proc_dir: str) -> Dict[str, str]:
    '''
    Crawls the processed dataset directory and builds a dictionary of the
    split names and their corresponding file paths.

    Parameters
    ----------
    proc_dir: str
        the directory containing the processed dataset

    Returns
    -------
    Dict[str, str]
        a dictionary containing the split names (eg 'test', 'val') and their
        corresponding file paths
    '''
    if not os.path.exists(proc_dir):
        raise ValueError(
            """Preprocessed dataset for artist not found. Please run
            preprocess_dataset() first.""")

    split_names = [os.path.splitext(file)[0]
                   for file in os.listdir(proc_dir)
                   if file.endswith(".json")]

    if not split_names:
        raise ValueError("No preprocessed datasets found.")

    split_dict = {split: os.path.join(proc_dir, f"{split}.json")
                  for split in split_names}

    return split_dict


def tokenize_dataset(artist: str) -> None:
    '''
    Tokenizes datasets and saves it to disk.

    Parameters
    ----------
    artist: str
        The name of the artist to tokenize the dataset for.
    '''
    artist_filename = f"{artist.replace(' ', '_').lower()}"
    artist_proc_dir = os.path.join(PROC_DIR, f"{artist_filename}")
    tokenized_dir = os.path.join(TOK_DIR, f"{artist_filename}")

    if os.path.exists(tokenized_dir):
        print(f"Tokenized dataset for {artist} already exists.")
        return

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    special_tokens = {
        "pad_token": tokenizer.eos_token,
        "additional_special_tokens": [
            "[Verse]",
            "[Hook]",
            "[Chorus]",
            "[Pre-Chorus]",
            "[Bridge]",
            "[Outro]",
            "[Intro]",
            "[Pre-Hook]",
            "[Post-Chorus]",
            "[Interlude]",
            "[Freestyle]",
            "[Post Commentary]",
            "[Skit]",
            "[Refrain]",
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.save_pretrained(tokenized_dir)

    data_files = build_split_dict(artist_proc_dir)
    dataset = load_dataset("json", data_files=data_files)

    if 'clean_lyrics' not in dataset['train'].column_names:
        raise ValueError("Clean lyrics column not found in dataset.")

    def tokenize_function(song):
        tokens = tokenizer(
            song["clean_lyrics"], padding="max_length", truncation=True)
        tokens['labels'] = tokens['input_ids'].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['clean_lyrics'])

    features = Features({
        "input_ids": Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int32")),
        "labels": Sequence(Value("int64")),
    })

    tokenized_dataset = tokenized_dataset.cast(features)

    os.makedirs(tokenized_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(tokenized_dir)
    print(f"Tokenized dataset for {artist} saved to disk.")
