from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments
)
from datasets import DatasetDict
from config import MODEL_DIR, TOK_DIR
import os


def fine_tune_model(artist: str) -> None:
    '''
    Fine-tunes the GPT2 model on the specified artist's lyrics.

    Parameters
    ----------
    artist: str
        The name of the artist to fine-tune the model on.
    '''
    artist_filename = f"{artist.replace(' ', '_').lower()}"
    tok_dir = os.path.join(
        TOK_DIR, f"{artist_filename}")
    model_file_path = os.path.join(MODEL_DIR, f"{artist_filename}_gpt2")

    if not os.path.exists(tok_dir):
        raise ValueError(
            f"""Tokenized dataset for {artist} not found. Please run
            tokenize_dataset() first.""")

    tokenizer = GPT2Tokenizer.from_pretrained(tok_dir)
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    model.resize_token_embeddings(len(tokenizer))

    dataset = DatasetDict.load_from_disk(tok_dir)
    train_dataset, val_dataset = dataset['train'], dataset['val']

    training_args = TrainingArguments(
        output_dir=model_file_path,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2
    )

    device = training_args.device.type
    print(f"Training model on {device}...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print(f"Fine-tuning model on {artist}...")
    trainer.train()
    model.save_pretrained(model_file_path)
    print(f"Model fine-tuning complete. Model saved to {model_file_path}")
