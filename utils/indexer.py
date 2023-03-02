import pickle
import sys
import os
import pprint
import time

from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer


class Indexer:
    def __init__(self, tfidf_path: str, words_index_path: str):
        self.tfidf_path = tfidf_path
        self.words_index_path = words_index_path

    def word_indexer(self, word: str):
        '''
        :return: relative path. eg. ../data/index/words_index/a/b/c/
        '''

        word_path = self.words_index_path
        for index in range(0, 3):
            if index > len(word):
                break

            word_path = os.path.join(word_path, word[index: index + 1])
        return word_path

    def link_indexer(self, locations_info: list, word: str):
        '''
        :param locations_info: All locations info we need to find.
                          eg. [[domain1, folder_name1], [domain2, folder_name2], ...]
        :return: The tf and idf socre of the word in every page it appears.
                 eg. {url1: [tf, idf], url2: [tf, idf]}
        '''
        link_info_s = {}
        urls = []
        tf_idf_scores = []
        tf_idf_paths = self.path_transformer(locations_info, word)
        url_paths = self.get_urls(locations_info)

        for url_path in url_paths:
            with open(url_path, 'rb') as f:
                urls.append(pickle.load(f))

        for tf_idf_path in tf_idf_paths:
            with open(tf_idf_path, 'rb') as f:
                tf_idf_scores.append(pickle.load(f))

        for index in range(len(urls)):
            link_info_s[urls[index]]  = tf_idf_scores[index]  # {url: [tf, idf]}

        return link_info_s

    def path_transformer(self, locations_info: list, word: str):
        '''
        :return: A list contains all path we need to find in link indexer.
                 eg. [tfidf/domain/url_folder_name/a/b/c/word.pickle, ....]
        '''
        word_path_in_link_index = word[0]
        paths = []
        for index in (range(1, 3) if len(word) >= 3 else range(1, len(word))):
            word_path_in_link_index = os.path.join(word_path_in_link_index, word[index])

        word_path_in_link_index = os.path.join(word_path_in_link_index, word + '.pickle')  # .../.../.../a/b/c/word.pickle

        for index in range(len(locations_info)):
            paths.append(os.path.join(self.tfidf_path, locations_info[index][0], locations_info[index][1],
                                            word_path_in_link_index))  # tfidf/domain/url_folder_name/a/b/c/word.pickle

        return paths

    def get_urls(self, locations_info:list):
        '''
        :return: Return the paths of the pickle file which contains url information.
        '''
        url_paths = []
        for info in locations_info:
            url_paths.append(os.path.join(self.tfidf_path, info[0], info[1], 'URL_info.pickle'))

        return url_paths

    def standardize_(self, query: str, stemmer=SnowballStemmer(language="english")):
        word_list = []
        unstandardized_words = word_tokenize(query.lower())
        for word in unstandardized_words:
            word = word.replace(' ', '')
            # Get the classification of words and do the initial filter
            word = stemmer.stem(word)
            word_list.append(word)
        return word_list

    def get_tf_idf(self, query: str):
        link_index_locations_info = []
        answer_pages_info = {}
        words = self.standardize_(query)
        for word in words:
            word_path = self.word_indexer(word)
            word_path = os.path.join(word_path, word + '.pickle')
            if not os.path.exists(word_path):
                continue

            with open(word_path, 'rb') as f:  # Open word index .pkl file to get link index location
                while True:
                    try:
                        location_info = pickle.load(f)  # [doamin, folder_name]
                        link_index_locations_info.append(location_info)
                    except:
                        break

            link_info = self.link_indexer(link_index_locations_info, word)
            answer_pages_info[word] = link_info

        # pprint.pprint(answer_pages_info)
