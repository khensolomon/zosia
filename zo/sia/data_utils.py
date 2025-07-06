# zo/sia/data_utils.py
#
# What it does:
# This module contains utility functions and classes for data preparation.
# It now includes functions to handle batching of data for more efficient training.

import torch
import re
import unicodedata

# ... (Lang class and normalize_string remain the same) ...
class Lang:
    """
    A class to manage the vocabulary of a language, including word-to-index
    mappings and special tokens.
    """
    def __init__(self, name, special_tokens):
        self.name = name
        self.special_tokens = special_tokens
        self.word2index = {token: i for i, token in enumerate(special_tokens.values())}
        self.word2count = {}
        self.index2word = {i: token for token, i in self.word2index.items()}
        self.n_words = len(special_tokens)

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicode_to_ascii(s):
    """Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    """Lowercase, trim, and remove non-letter characters"""
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip()

def prepare_data(pairs, special_tokens):
    """
    Normalizes raw text pairs, then creates Lang instances for input and
    output languages and builds the vocabulary.
    """
    print("Normalizing strings and building vocabulary...")
    normalized_pairs = [[normalize_string(p[0]), normalize_string(p[1])] for p in pairs]

    input_lang = Lang("zolai", special_tokens)
    output_lang = Lang("english", special_tokens)
    
    for pair in normalized_pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
        
    return input_lang, output_lang, normalized_pairs

def calculate_max_length(pairs):
    """Calculates the maximum sentence length from all data pairs."""
    max_len = 0
    for input_sentence, output_sentence in pairs:
        max_len = max(max_len, len(input_sentence.split(' ')) + 1) # +1 for EOS
        max_len = max(max_len, len(output_sentence.split(' ')) + 1) # +1 for EOS
    return max_len

def tensor_from_sentence(lang, sentence, max_len, device):
    """Converts a single sentence into a padded tensor of word indices."""
    indexes = [lang.word2index.get(word, lang.word2index[lang.special_tokens.unk]) for word in sentence.split(' ')]
    indexes.append(lang.word2index[lang.special_tokens.eos])
    
    if len(indexes) > max_len:
        indexes = indexes[:max_len]
    else:
        while len(indexes) < max_len:
            indexes.append(lang.word2index[lang.special_tokens.pad])
        
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensors_from_pair(pair, input_lang, output_lang, max_length, device):
    """Converts a single pair of sentences to tensors."""
    input_tensor = tensor_from_sentence(input_lang, pair[0], max_length, device)
    target_tensor = tensor_from_sentence(output_lang, pair[1], max_length, device)
    return (input_tensor, target_tensor)
