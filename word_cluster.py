import jieba
import gensim
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
plt.rcParams['font.sans-serif'] = ['SimHei']    # 指定默认字体：解决plot不能显示中文问题
plt.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

#从gensim以中文维基百科训练好的word2vec词向量抽取出来作为relate.csv的词向量
def get_save_vector():
    with open('./data/relate.csv', 'r',encoding='utf-8') as f:
        text = f.read().split('\n')
        text = [list(jieba.cut(i)) for i in text]
        tmp = []
        for i in text:
            tmp.extend(i)
        del text
    tmp = set(tmp)
    model = gensim.models.KeyedVectors.load_word2vec_format('./store/w2v_vector.bin', binary = False)
    wordlist = []
    vectorlist = []
    for item in tmp:
        try:
            vector=model.get_vector(item)
        except:
            continue
        wordlist.append(item)
        vectorlist.append(vector)
    embending = dict(zip(wordlist, vectorlist))
    del wordlist, vectorlist
    with open('./store/word_embending', 'wb') as f:
        pickle.dump(embending, f)

def set_list(textlist):
    text = list(set(textlist))
    text.sort(key=textlist.index)
    return text

def relate_deal():
    with open('./store/word_embending', 'rb') as f:
        embending = pickle.load(f)
    wordlist = list(embending.keys())
    with open('./data/stopword.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read().split('\n')
    with open('./data/relate.csv', 'r', encoding='utf-8') as f:
       text = f.read().replace(' ','').split('\n')
    text = [i for i in text if len(i) < 9]
    text = [list(jieba.cut(item)) for item in text]
    sentence = [' '.join([j for j in i if (j not in stopwords)]) for i in text]
    sentence = set_list(sentence)
    vector = [[j for j in i.split(' ') if (j in wordlist)] for i in sentence]
    for i, val in enumerate(vector):
        if vector[i] == []:
            del sentence[i],vector[i]
    sentence = [i.replace(' ','') for i in sentence]
    with open('./store/sentence', 'wb') as f:
        pickle.dump(sentence, f)
    with open('./store/vector', 'wb') as f:
        pickle.dump(vector, f)

def get_save_sentence_embending():
    with open('./store/word_embending', 'rb') as f:
        embending = pickle.load(f)
    with open('./store/vector', 'rb') as f:
        textvector = pickle.load(f)
    textvector = [np.array([embending[j] for j in i]) for i in textvector]
    textvector = [np.mean(i, axis=0) for i in textvector]
    low_dim_embs = TSNE(learning_rate=100).fit_transform(textvector)
    with open('./store/low_dim_word_embending', 'wb') as f:
        pickle.dump(low_dim_embs, f)

def draw():
    with open('./store/sentence', 'rb') as f:
        textsentence = pickle.load(f)
    with open('./store/low_dim_word_embending', 'rb') as f:
        low_dim_embs = pickle.load(f)
    figure = plt.figure(figsize=(100,100))
    db = DBSCAN(eps=3, min_samples=12).fit_predict(low_dim_embs)
    x=low_dim_embs[:, 0]
    y=low_dim_embs[:, 1]
    plt.scatter(x, y, c=db)
    for i, word in enumerate(textsentence):
        x, y = low_dim_embs[i, :]
        plt.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    figure.savefig('./picture/log')
    plt.close(figure)

def show():
    img = Image.open('./picture/log.png')
    img.show()

#如果你有训练好的中文维基百科的word2vec词向量，放到store文件夹，命名为w2v_vector.bin，去掉注释，没有的话别去掉
#get_save_vector()

relate_deal()
get_save_sentence_embending()
draw()
show()