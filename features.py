# features.py

import numpy as np

def text_length_func(X):
    return np.array([[len(text.split())] for text in X])

def unique_words_func(X):
    return np.array([[len(set(text.split()))] for text in X])

def avg_word_length_func(X):
    return np.array([[np.mean([len(w) for w in text.split()]) if len(text.split()) > 0 else 0] for text in X])

def sentence_count_func(X):
    return np.array([[text.count('.') + text.count('!') + text.count('?')] for text in X])
