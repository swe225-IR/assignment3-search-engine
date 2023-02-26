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
        :return: relative path. eg. ../data/index/words_index/a/b/c/abc.pickle
        '''
        word_path = self.words_index_path
        for index in range(0, 3):
            if index > len(word):
                break

            word_path = os.path.join(word_path, word[index: index + 1])

        return word_path

    def link_indexer(self, paths: list, word: str):
        '''
        :return: The tf and idf socre of the word in every page it appears.
                 eg. {url1: [tf, idf], url2: [tf, idf]}
        '''
        link_info_s = {}
        for path in paths:
            with open(path, 'rb') as f:
                link_info = pickle.load(f)
                for k in link_info:
                    url = k
                    break
                link_info_s[url] = link_info[url].get(word, None)
        return link_info_s

    def standardize_(self, query: str, stemmer=SnowballStemmer(language="english")):
        word_list = []
        unstandardized_words = word_tokenize(query.lower())
        for word in unstandardized_words:
            word = word.replace(' ', '')
            # Get the classification of words and do the initial filter
            word = stemmer.stem(word)
            word_list.append(word)
        return word_list

    def get_urls(self, query: str):
        answer_pages_info = {}
        words = self.standardize_(query)
        for word in words:
            path = self.word_indexer(word)

            path = os.path.join(path, word + '.pickle')
            paths = list()

            if not os.path.exists(path):
                continue

            with open(path, 'rb') as f:
                while True:
                    try:
                        link_id = pickle.load(f)
                        link_sub_path = self.tfidf_path + link_id[0] + '/' + link_id[1]
                        paths.append(link_sub_path + '.pickle')
                    except:
                        break

            link_info = self.link_indexer(paths, word)
            answer_pages_info[word] = link_info

        # pprint.pprint(answer_pages_info)
