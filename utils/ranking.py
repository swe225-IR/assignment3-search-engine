from typing import Dict
import math
import json


class TFIDF:
    def __init__(self, word_frequencies: Dict[str, int], word_count: int, corpus: Dict[str, Dict], corpus_count: int):
        self.word_features = None
        self.word_frequencies = word_frequencies
        self.word_counts = word_count
        self.corpus = corpus
        self.corpus_count = corpus_count
        self.word_tf = {}
        self.word_idf = {}
        self.word_if_idf = {}

    def calculate_tf(self):
        for k, v in self.word_frequencies.items():
            self.word_tf[k] = v * 1.0 / self.word_counts

    def calculate_idf(self):
        for k, v in self.word_frequencies.items():
            num_of_corpus_contain_k = len(self.corpus[k])
            self.word_idf[k] = math.log(self.corpus_count / (num_of_corpus_contain_k + 1))

    def calculate_if_idf(self):
        self.calculate_tf()
        self.calculate_idf()
        for k, v in self.word_tf.items():
            self.word_if_idf[k] = v * self.word_idf[k]

        self.word_features = sorted(self.word_if_idf.items(), key=lambda x: x[1], reverse=True)

    def output_page_feature_words(self):
        fp = open("test.json", 'a')
        json.dump(self.word_features, fp, indent=4)
