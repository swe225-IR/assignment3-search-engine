from typing import Dict
import math
import json
import numpy as np
import os
import pickle
import sys


class TFIDF:
    def __init__(self,
                 corpus: Dict[str, int],
                 page_number: int,
                 weights=[1, 4, 7, 8, 9, 10, 15]):

        self.corpus = corpus
        self.page_number = page_number
        self.weights = np.array(weights)
        self.word_counts = 0
        self.word_tf = dict()
        self.word_idf = dict()
        self.word_tf_idf = dict()
        self.tf_idf_result = dict()  # {1st of word: {2nd of word: {3rd of word: {word: [tf, idf]}}}}

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

    def save(self, tfidf_folder_path, domain_folder, sub_folder_name, url:str):
        '''
        :param tfidf_folder_path: '../data/index/tfidf/'
        :param domain_folder: xxx_xxx_xxx_edu
        :param sub_folder_name: xxxxx
        '''
        if not os.path.exists(tfidf_folder_path):
            os.mkdir(tfidf_folder_path)

        if not os.path.exists(tfidf_folder_path + domain_folder):  # '../data/index/tfidf/xxx_xxx_xxx_edu/'
            os.mkdir(tfidf_folder_path + domain_folder)

        page_folder_path = os.path.join(tfidf_folder_path + domain_folder, sub_folder_name) # '../data/index/tfidf/xxx_xxx_xxx_edu/xxxxx/'
        if not os.path.exists(page_folder_path):
            os.mkdir(page_folder_path)

        url_file_path = os.path.join(page_folder_path, 'URL_info.pickle')
        with open(url_file_path, 'wb') as f:
            pickle.dump(url, f)

        for k, v in self.tf_idf_result.items():
            if not k.isalnum():
                continue

            path = page_folder_path
            # Create the folder with tree structure for a word
            for index in (range(3) if len(k) >= 3 else range(len(k))):
                path = os.path.join(path, k[index])
                if not os.path.exists(path):
                    os.mkdir(path)

            path = os.path.join(path, k + '.pickle')
            with open(path, 'wb') as f:
                pickle.dump(self.tf_idf_result[k], f)

    def reset(self):
        self.word_counts = 0
        self.word_tf = dict()
        self.word_idf = dict()
        self.word_tf_idf = dict()
        self.tf_idf_result = dict()


def read_json(path):
    with open(path, 'r') as f:
        page = json.load(f)
        for key in page:
            url = key
            break
        return [url, page[[*page][0]]['word_frequency_weights']]


def calculated_finished_process(root_dir, output_dir):
    page_number = 0
    sub_folders = os.listdir(output_dir)[:-1]  # Ignore final folder because we cannot make sure the folder is finished
    for index in range(len(sub_folders)):
        temp_path = os.path.join(root_dir, sub_folders[index])
        page_number += len(os.listdir(temp_path))
        sub_folders[index] = temp_path

    return [page_number, sub_folders]


if __name__ == '__main__':
    is_restart = sys.argv[1]
    print('Loading corpus...')
    with open('../data/index/corpus.json', 'r') as f:
        corpus = json.load(f)

    with open('../data/index/global_info.json', 'r') as f:
        page_number = json.load(f)

    root_dir = '../METADATA/'
    output_dir = '../data/index/tfidf/'

    print('Start calculating...')
    page_progress = 0
    finished_folders = []
    tfidf_calculator = TFIDF(corpus, page_number)

    if is_restart == '0':
        [page_progress, finished_folders] = calculated_finished_process(root_dir, output_dir)
        print('    |--Start from page: ' + str(page_progress))

    for dir_, _, files in os.walk(root_dir):
        if is_restart == '0' and dir_ in finished_folders:
            print('        |--Done: ' + dir_ + ' ' + str(int((page_progress / page_number) * 100)) + '%')
            continue

        for file_name in files:
            rel_dir = os.path.relpath(dir_, root_dir)  # dir_: xxx_xxx_xxx_edu
            rel_file = os.path.join(rel_dir, file_name)  # file_name: xxx_meta_data.json
            [url, word_frequencies] = read_json(root_dir + rel_file)

            tfidf_calculator.calculate(word_frequencies)
            tfidf_calculator.save(tfidf_folder_path=output_dir,
                                  domain_folder=rel_dir,
                                  sub_folder_name=file_name[:-15],
                                  url=url)
            tfidf_calculator.reset()
            page_progress += 1

        print('        |--Done: ' + dir_ + ' ' + str(int((page_progress / page_number) * 100)) + '%')

