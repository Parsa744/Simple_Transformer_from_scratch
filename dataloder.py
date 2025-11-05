import pandas as pd
from sklearn.model_selection import train_test_split
import gensim.downloader as api
import numpy as np

def clean_sentence(sent,seq_len=32,placeholder_token='<pad>',start_token='<start>',has_start_token=False):
    sent = str(sent).lower()
    if has_start_token:
        sent = start_token + ' ' + sent
    sent_list = []
    for word in sent.split():
        for char in word:
            if char.isalnum():
                word = ''.join(char for char in word if char.isalnum())
        sent_list.append(word)
    if len(sent_list) > seq_len:
        sent_list = sent_list[:seq_len]
    elif len(sent_list) < seq_len:
        sent_list.extend([placeholder_token]*(seq_len-len(sent_list)))
    return sent_list


def data2vec(de_wv,eng_wv, data_df,seq_len=32):
    de_list = []
    for sent in data_df['de']:
        words = clean_sentence(sent,seq_len,'punkt')
        sent_list = []
        for word in words:
            try:
                vec = de_wv[word]
            except KeyError:
                vec = de_wv['unknown']
            sent_list.append(vec)
        end_word = 'punkt'
        vec = de_wv[end_word]
        sent_list.append(vec)
        de_list.append(sent_list)

    en_list = []
    for sent in data_df['en']:
        words = clean_sentence(sent,seq_len,'point')
        sent_list = []
        for word in words:
            try:
                vec = eng_wv[word]
            except KeyError:
                vec = eng_wv['unknown']
            sent_list.append(vec)
        end_word = 'point'
        vec = eng_wv[end_word]
        sent_list.append(vec)
        en_list.append(sent_list)
    return pd.DataFrame([de_list,en_list]).T


def vec2eng(vec_sentence):
    eng_wv = api.load('word2vec-google-news-300')
    most_similar = eng_wv.similar_by_vector(vec_sentence.detach().cpu().numpy(), topn=3)
    return most_similar


def vec2de(vec_sentence):
    de_wv = api.load('fasttext-wiki-news-subwords-300')
    most_similar = de_wv.similar_by_vector(vec_sentence.detach().cpu().numpy(), topn=3)
    return most_similar


def add_start_token_2_dataset(sentences, start_token='<start>'):
    eng_wv = api.load('word2vec-google-news-300')
    sentences = sentences.pop(0)
    print('sentences',sentences[0])
    for i in range(len(sentences)):
        sentence = sentences[i]
        start_token = eng_wv['start']
        sentence.insert(0, start_token)

    print('update',sentences[0])
    return sentences


def add_start_token(sentence_org, start_token='<start>'):
    sentence = sentence_org.copy()
    eng_wv = api.load('word2vec-google-news-300')

    # Get the start token vector and reshape it to match the sentence dimensions
    start_token_vec = eng_wv['start'].reshape(1, -1)  # Shape: (1, 300)

    # Insert at the beginning along axis 0
    result = np.insert(sentence, 0, start_token_vec, axis=0)

    return result


def load_data():
    #x = clean_sentence('Hello, world! This is a test-sentence.',has_start_token=True)


    de_wv = api.load('fasttext-wiki-news-subwords-300')
    eng_wv = api.load('word2vec-google-news-300')

    '''
    try:
        vec_koenig = de_wv['.']
    except KeyError:
        vec_koenig = de_wv['point']
    print(vec_koenig)

    try:
        vec_koenig = eng_wv['.']
    except KeyError:
        vec_koenig = eng_wv['point']
    print(vec_koenig)
    '''

    #vec_king = wv['King']
    #print(vec_king)

    data_df = pd.read_csv('deu.txt', sep='\t', usecols=[0, 1])
    data_df.columns = ['en', 'de']
    data_df.head()

    data_df = data2vec(de_wv, eng_wv, data_df)

    train_df, valid_df = train_test_split(data_df, test_size=0.1, shuffle=True, random_state=28)

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    #print("traing shape",train_df.shape)
    #print("test shape",valid_df.shape)

    return train_df, valid_df
def test():
    #x = clean_sentence('Hello, world! This is a test-sentence.',has_start_token=True)


    de_wv = api.load('fasttext-wiki-news-subwords-300')
    eng_wv = api.load('word2vec-google-news-300')


    data_df = pd.read_csv('deu.txt', sep='\t', usecols=[0, 1])
    data_df.columns = ['en', 'de']
    data_df = data_df.head()

    data_df = data2vec(de_wv, eng_wv, data_df)

    train_df, valid_df = train_test_split(data_df, test_size=0.1, shuffle=True, random_state=28)

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    train_df = add_start_token(train_df, start_token='start', model=de_wv)
    valid_df = add_start_token(valid_df, start_token='start', model=de_wv)


    return train_df, valid_df
load_data()