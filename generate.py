import argparse
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from config import MODEL_DIR, TOK_DIR


def load_model_and_tokenizer(artist: str):
    model_path = os.path.join(MODEL_DIR, "riff_raff_gpt2")
    tokenizer_path = os.path.join(TOK_DIR, "riff_raff")

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    return model, tokenizer


def generate_lyrics(artist: str, prompt: str, max_length=100):
    model, tokenizer = load_model_and_tokenizer("riff_raff")

    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs,
                            max_length=max_length,
                            num_return_sequences=1,
                            temperature=0.5,
                            top_p=0.9,
                            repetition_penalty=1.3,
                            no_repeat_ngram_size=3,
                            do_sample=True)

    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate lyrics using a fine-tuned GPT2 model.")
    parser.add_argument("-a", "--artist", type=str, required=True,
                        help="The name of the artist to generate lyrics for")
    parser.add_argument("-p", "--prompt", type=str, required=True,
                        help="The prompt to generate lyrics from")
    args = parser.parse_args()

    print(generate_lyrics(args.artist, args.prompt))
