from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def prepare_tinystories():
    logging.info("Downloading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories")

    logging.info("Training BPE tokenizer...")
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(vocab_size=10000, special_tokens=["</s>"])
    tokenizer.train_from_iterator(dataset["train"]["text"], trainer)
    tokenizer.save("bpe-tokenizer_tinystories.json")

    logging.info("Tokenizing data...")
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe-tokenizer_tinystories.json", pad_token="</s>")
    def tokenize(text):
        return fast_tokenizer.encode(text, return_tensors="pt").squeeze()

    train_ids = torch.cat([tokenize(text) for text in dataset["train"]["text"]])
    valid_ids = torch.cat([tokenize(text) for text in dataset["validation"]["text"]])

    logging.info("Saving tokenized data...")
    torch.save(train_ids, "tokenized-train-samples_vocab-10k.pt")
    torch.save(valid_ids, "tokenized-valid-samples_vocab-10k.pt")
    logging.info("Dataset preparation completed.")

if __name__ == "__main__":
    prepare_tinystories()