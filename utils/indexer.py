import pickle
import os
import pprint
import hashlib
import sys
import time

from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer


class Indexer:
    def __init__(self, indexer_folder: str):
        self.indexer_folder = indexer_folder

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
        answer_pages_info = {}
        words = self.standardize_(query)

        for word in words:
            word_hash = hashlib.md5(word.encode()).hexdigest()
            word_path = os.path.join(self.indexer_folder, word_hash + '.pickle')
            if not os.path.exists(word_path):
                continue

            if answer_pages_info.get(word, None) is None:
                answer_pages_info[word] = []

            with open(word_path, 'rb') as f:  # Open word index .pkl file to get link index location
                while True:
                    try:
                        single_page_info = pickle.load(f)  # [doamin, folder_name]
                        answer_pages_info[word].append(single_page_info)
                    except:
                        break

        # pprint.pprint(answer_pages_info)


if __name__ == '__main__':
    start_time = time.time()
    query = sys.argv[1]
    indexer = Indexer(indexer_folder='../data/index/indexer/')
    indexer.get_tf_idf(query)
    total_time = time.time() - start_time
    print('Time in total: ' + format(total_time, '.4f'))