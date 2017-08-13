# http://qiita.com/takumi_TKHS/items/4a56ac151c60da8bde4b
import sys
import numpy as np
import fasttext as ft
from scipy.spatial import distance
from gensim.models.word2vec import Word2Vec

def word2id(bow, word_id):
    for w in bow:
        if word_id.has_key(w) == False:
            word_id[w] = len(word_id)

    return word_id

def compute_tf(sentences, word_id):
    tf = np.zeros([len(sentences), len(word_id)])

    for i in range(len(sentences)):
        for w in sentences[i]:
            tf[i][word_id[w]] += 1

    return tf

def compute_df(sentences, word_id):
    df = np.zeros(len(word_id))

    for i in range(len(sentences)):
        exist = {}
        for w in sentences[i]:
            if exist.has_key(w) == False:
                df[word_id[w]] += 1
                exist[w] = 1
            else:
                continue

    return df

def compute_idf(sentences, word_id):
    idf = np.zeros(len(word_id))
    df = compute_df(sentences, word_id)

    for i in range(len(df)):
        idf[i] = np.log(len(sentences)/df[i]) + 1

    return idf

def compute_tfidf(sentences):
    word_id = {}

    for sent in sentences:
        word_id = word2id(sent, word_id)

    tf = compute_tf(sentences, word_id)
    idf = compute_idf(sentences, word_id)

    tf_idf = np.zeros([len(sentences), len(word_id)])

    for i in range(len(sentences)):
        tf_idf[i] = tf[i] * idf

    return tf_idf

def compute_cosine(v1, v2):
    return 1 - distance.cosine(v1, v2)

def sent2vec(bow, model_w):
    vector = np.zeros(50)
    N = len(bow)

    for b in bow:
        try:
            vector += model_w[b]
        except:
            continue

    vector = vector / float(N)

    return vector

def compute_word2vec(sentences):
    # https://github.com/shiroyagicorp/japanese-word2vec-model-builder
    model_w = Word2Vec.load('./word2vec.gensim.model')
    vector = np.zeros([len(sentences), 50])

    for i in range(len(sentences)):
        vector[i] = sent2vec(sentences[i], model_w)

    return vector

def PowerMethod(CosineMatrix, N, err_tol):
    p_old = np.array([1.0/N]*N)
    err = 1

    while err > err_tol:
        err = 1
        p = np.dot(CosineMatrix.T, p_old)
        err = np.linalg.norm(p - p_old)
        p_old = p

    return p

def lexrank(sentences, threshold, vectorizer):
    N = len(sentences)
    CosineMatrix = np.zeros([N, N])
    degree = np.zeros(N)
    L = np.zeros(N)

    if vectorizer == "tf-idf":
        vector = compute_tfidf(sentences)
    elif vectorizer == "word2vec":
        vector = compute_word2vec(sentences)

    # Computing Adjacency Matrix
    for i in range(N):
        for j in range(N):
            CosineMatrix[i,j] = compute_cosine(vector[i], vector[j])
            if CosineMatrix[i,j] > threshold:
                CosineMatrix[i,j] = 1
                degree[i] += 1
            else:
                CosineMatrix[i,j] = 0

    # Computing LexRank Score
    for i in range(N):
        for j in range(N):
            CosineMatrix[i,j] = CosineMatrix[i,j] / degree[i]

    L = PowerMethod(CosineMatrix, N, err_tol=10e-6)

    return L

def main():
    sentences = [l.rstrip() for l in sys.stdin.readlines() if len(l.rstrip())>0]
    vec = list(lexrank(sentences, 0.3, "word2vec"))
    res = sorted([ [r, sentences[i]] for i, r in enumerate(vec) ], key=lambda a:a[0])[::-1]
    out = "".join([s for r,s in res[0:3]])
    print(out)

if __name__ == '__main__':
    main()
