
import re

def word_tokenizer(sentence):

    sentence = re.sub(r'https://[^ ]+', '<URL>', sentence)
    sentence = re.sub(r' @[^ ]+', ' <MENTION>', sentence)
    sentence = re.sub(r'[^ ]+@[^ ]+', '<MAILID>', sentence)
    sentence = re.sub(r'#[^ ]+', '<HASHTAG>', sentence)
    sentence = re.sub(r'\b\d+\b', '<NUM>', sentence)
    sentence = re.sub(r',', ' , ', sentence)
    sentence = re.sub(r"'([^']+)'", r" '\1' ", sentence)
    sentence = re.sub(r'"([^"]+)"', r' "\1" ', sentence)
    sentence = re.sub(r"([a-zA-Z])'([a-zA-Z])", r'\1 \2', sentence)
    sentence = re.sub(r'\.$', ' . ', sentence)
    tokens = sentence.split()
    return tokens

def tokenize(paragraph):
    sentences = re.split(r'(?<=\.|\?)\s', paragraph)
    tokenized_sentences = [word_tokenizer(sentence) for sentence in sentences]
    return tokenized_sentences



