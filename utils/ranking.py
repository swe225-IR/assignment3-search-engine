# %%
from typing import Dict
import math
import json
import numpy as np
import os
from collections import Counter

class TFIDF:
    def __init__(self, word_frequencies: Dict[str, int], corpus: Dict[str, int], weights: list):
        self.word_frequencies = word_frequencies
        self.word_counts = 0
        self.corpus = corpus
        self.corpus_count = self.corpus['total page number']
        self.weights = np.array(weights)
        self.word_tf = {}
        self.word_idf = {}
        self.word_if_idf = {}
    
    def calculate_weighted_frequency(self, k, v):
        v = np.array(v)
        v[0] -= np.sum(v[1:])
        weighted_freq = v.dot(self.weights)
        self.word_tf[k] = weighted_freq
        self.word_counts += weighted_freq

    def calculate_tf(self):
        for k, v in self.word_frequencies.items():
            self.calculate_weighted_frequency(k, v)
        for k in self.word_tf.keys():
            self.word_tf[k] /= self.word_counts

    def calculate_idf(self):
        for k in self.word_frequencies.keys():
            num_of_corpus_contain_k = self.corpus[k]
            self.word_idf[k] = math.log(self.corpus_count / (num_of_corpus_contain_k + 1))

    def calculate_if_idf(self):
        self.calculate_tf()
        self.calculate_idf()
        for k, v in self.word_tf.items():
            self.word_if_idf[k] = v * self.word_idf[k]

        self.word_features = Counter(self.word_if_idf)

    def output_page_feature_words(self, fpath):
        if not os.path.exists(fpath.split('\\')[0]):
            os.mkdir(fpath.split('\\')[0]) 
        with open(fpath, 'w') as f:
            json.dump(self.word_features, f, indent=4)

# %%
if __name__ == '__main__':
    with open('../corpus.json', 'r') as f:
        corpus = json.load(f)

    root_dir = 'D:/UCI/1-Q2/CS221/Assignment-3-web-crawler/assignment3-search-engine/METADATA/'
    output_dir = 'D:/UCI/1-Q2/CS221/Assignment-3-web-crawler/assignment3-search-engine/tfidf/'
    weights = [1, 4, 7, 8, 9, 10, 15]

    for dir_, _, files in os.walk(root_dir):
        for file_name in files:
            rel_dir = os.path.relpath(dir_, root_dir)
            rel_file = os.path.join(rel_dir, file_name)
            with open(root_dir + rel_file, 'r') as f:
                page = json.load(f)
            TFIDF_calc = TFIDF(page[[*page][0]]['word_frequency_weights'], corpus, weights)
            TFIDF_calc.calculate_if_idf()
            TFIDF_calc.output_page_feature_words(output_dir + rel_file)
#%%