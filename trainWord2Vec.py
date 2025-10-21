from models import torchRNN
import torch
import torch.nn as nn
import torch.optim as optim
import os
import urllib.request
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from gensim.models import Word2Vec

nltk.download('punkt_tab')

def trainWord2Sent(text_path = 'shakespeare.txt'):
    text = open(text_path).read()
    tokenize_sentences = sent_tokenize(text)
    tokenize_words = []
    for sent in tokenize_sentences:
        tokenize_words.append(word_tokenize(sent))
    missing_word = ['<unk>']
    pading = ['<pad>']
    tokenize_words.append(pading)
    tokenize_words.append(missing_word)

    model = Word2Vec(sentences=tokenize_words, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")




def main():
    trainWord2Sent()

if __name__ == "__main__":
    main()

