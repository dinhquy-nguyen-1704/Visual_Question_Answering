from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import spacy

def get_tokens(data_iter):
    eng = spacy.load("en_core_web_sm")
    for sample in data_iter:
        question = sample["question"]
        yield [token.text for token in eng.tokenizer(question)]

def tokenize(question, max_sequence_length, vocab):
    eng = spacy.load("en_core_web_sm")
    tokens = [token.text for token in eng.tokenizer(question)]
    sequence = [vocab[token] for token in tokens]
    if len(sequence) < max_sequence_length:
        sequence += [vocab['<pad>']]*(max_sequence_length - len(sequence))
    else:
        sequence = sequence[:max_sequence_length]

    return sequence
