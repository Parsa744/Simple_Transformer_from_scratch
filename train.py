import numpy as np
import matplotlib.pyplot as plt
from models import torchRNN,seq2oneRNN
import torch
import torch.nn as nn
import torch.optim as optim
import os
import urllib.request
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from gensim.models import Word2Vec
import random


nltk.download('punkt_tab')

def text2vec(text_path = 'shakespeare.txt',sent_size = 32):
    text = open(text_path).read()
    tokenize_sentences = sent_tokenize(text)
    model = Word2Vec.load('word2vec.model')


    text_vector_list = []

    for sent in tokenize_sentences:
        sent_list = []
        word_list = word_tokenize(sent)
        missing_word = random.choice(word_list)
        index = word_list.index(missing_word)
        word_list[index] = '<unk>'

        for word_index in range(sent_size):
            if word_index < len(word_tokenize(sent)):
                word = word_list[word_index]
            else:
                word = '<pad>'
            sent_list.append(model.wv[word])

        text_vector_list.append([sent_list,model.wv[missing_word]])

    #print(len(text_vector_list[3]))
    return text_vector_list



def main():

    list_of_vectors = text2vec()
    hid_size = 100

    myRNN = seq2oneRNN(100,hid_size,100)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(myRNN.parameters(), lr=0.1)
    total_loss = 0
    H0 = torch.zeros(hid_size, 1)  # Initialize as tensor with correct shape
    loss_list = []
    for sentence in list_of_vectors:
        optimizer.zero_grad()
        input = sentence[0]
        missing_word = sentence[1]
        pred, _ = myRNN.forward_for_seq(input_seq=input, hidden=H0)
        missing_word = torch.FloatTensor(missing_word).unsqueeze(1)


        loss = criterion(pred, missing_word)

        #print(loss)
        total_loss+=loss
        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()
        pred_word = Word2Vec.load('word2vec.model').wv.most_similar(positive=[pred.squeeze().detach().numpy()], topn=1)
        print('pred_word',pred_word)
        print('missing_word',Word2Vec.load('word2vec.model').wv.most_similar(positive=[missing_word.squeeze().detach().numpy()], topn=1))
    plt.ylabel('loss')
    plt.show()

if __name__ == "__main__":
    main()











    '''def main():
    text2List()



    x1 = [1,2,2]
    Y1 = [0,2,0,4]
    H0 = [0,0,0,0]
    x2 = [1,2,4]
    Y2 = [3,0,1,1]
    x3 = [3,0,5]
    Y3 = [5,2,7,3]
    x4 = [5,6,7]
    Y4 = [3,4,9,2]
    myRNN = torchRNN(3,4,4)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(myRNN.parameters(), lr=0.1)

    Yp1,H1 = myRNN.forward(input=x1,hidden=H0)

    loss = criterion(Yp1,torch.FloatTensor(data=Y1).unsqueeze(1))
    optimizer.zero_grad()

    print('loss',loss)

    Yp2,H2 = myRNN.forward(input=x2,hidden=H1)


    loss = criterion(Yp2,torch.FloatTensor(data=Y2).unsqueeze(1))

    print('loss',loss)

    Yp3,H3 = myRNN.forward(input=x3,hidden=H2)

    loss = criterion(Yp3,torch.FloatTensor(data=Y3).unsqueeze(1))

    print('loss',loss)


    Yp4,H4 = myRNN.forward(input=x4,hidden=H3)


    loss = criterion(Yp4,torch.FloatTensor(data=Y4).unsqueeze(1))
    loss.backward()
    optimizer.step()
    print('loss',loss)
'''
