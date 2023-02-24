# %%
from typing import Dict
import math
import json
import numpy as np
import os
from collections import Counter


class TFIDF:
    def __init__(self,
                 corpus: Dict[str, int],
                 page_number: int,
                 weights=[1, 4, 7, 8, 9, 10, 15]):

        self.corpus = corpus
        self.page_number = page_number
        self.weights = np.array(weights)
        self.word_counts = 0
        self.word_tf = {}
        self.word_idf = {}
        self.word_tf_idf = {}
        self.tf_idf_result = {}  # {word: [tf, idf]}

    def calculate_weighted_frequency(self, k, v):
        v = np.array(v)
        v[0] -= np.sum(v[1:])
        weighted_freq = v.dot(self.weights)
        self.word_tf[k] = weighted_freq
        self.word_counts += weighted_freq

    def calculate_tf(self, word_frequencies):
        for k, v in word_frequencies.items():
            self.calculate_weighted_frequency(k, v)

        for k in self.word_tf.keys():
            self.word_tf[k] /= self.word_counts

            self.tf_idf_result[k] = [0, 0]
            self.tf_idf_result[k][0] = self.word_tf[k]

    def calculate_idf(self, word_frequencies):
        for k in word_frequencies.keys():
            num_of_corpus_contain_k = self.corpus[k]
            self.word_idf[k] = math.log(self.page_number / (num_of_corpus_contain_k + 1))

            self.tf_idf_result[k][1] = self.word_idf[k]

    def calculate_if_idf(self):
        for k, v in self.word_tf.items():
            self.word_tf_idf[k] = v * self.word_idf[k]

    def calculate(self, word_frequencies):
        self.calculate_tf(word_frequencies)
        self.calculate_idf(word_frequencies)
        # self.calculate_if_idf()

    def save(self, folder_path, sub_folder, file_name, url):
        if not os.path.exists(folder_path + sub_folder):
            os.mkdir(folder_path + sub_folder)

        save_dic = {url: self.tf_idf_result}
        with open(folder_path + file_name, 'w') as f:
            json.dump(save_dic, f, indent=4)

    def reset(self):
        self.word_counts = 0
        self.word_tf = {}
        self.word_idf = {}
        self.word_tf_idf = {}
        self.tf_idf_result = {}


def read_json(path):
    with open(path, 'r') as f:
        page = json.load(f)
        for key in page:
            url = key
            break

        return [url, page[[*page][0]]['word_frequency_weights']]


if __name__ == '__main__':
    print('Loading corpus...')
    with open('../data/index/corpus.json', 'r') as f:
        corpus = json.load(f)

    with open('../data/index/global_info.json', 'r') as f:
        page_number = json.load(f)

    root_dir = '../METADATA/'
    output_dir = '../data/index/tfidf/'

    print('Start calculating...')
    page_progress = 0
    tfidf_calculator = TFIDF(corpus, page_number)
    for dir_, _, files in os.walk(root_dir):
        for file_name in files:
            rel_dir = os.path.relpath(dir_, root_dir)
            rel_file = os.path.join(rel_dir, file_name)
            [url, word_frequencies] = read_json(root_dir + rel_file)

            tfidf_calculator.calculate(word_frequencies)
            tfidf_calculator.save(output_dir, rel_dir, rel_file, url)
            tfidf_calculator.reset()

            page_progress += 1
            if page_progress % 500 == 0:
                print('Progress: ' + str(int((page_progress / page_number) * 100)) + '%')
