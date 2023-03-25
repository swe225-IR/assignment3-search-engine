import pickle
import os
import pprint
import hashlib
import sys
import json
import time
import math

from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.text import TextCollection


class Indexer:
    def __init__(self, indexer_folder: str, query: str):
        self.indexer_folder = indexer_folder
        # self.query = query
        self.words = self.standardize_(query)
        # self.query_tf_idf = {}
        self.pages_reverse_tfidf_rank = {}  # {word: {url: relative_rank, ...}, word2: ...} sorted from low to high
        self.pages_reverse_page_rank = {}  # {url: relative_rank, ...} sorted from low to high
        self.final_rank = []  # Final result
        self.all_pages = set()  # Record all page we need to rank

    def standardize_(self, query: str, stemmer=SnowballStemmer(language="english")):
        word_list = []
        unstandardized_words = word_tokenize(query.lower())
        for word in unstandardized_words:
            word = word.replace(' ', '')
            # Get the classification of words and do the initial filter
            word = stemmer.stem(word)
            word_list.append(word)
        return word_list

    def get_tf_idf(self):
        for word in self.words:
            word_hash = hashlib.md5(word.encode()).hexdigest()
            word_path = os.path.join(self.indexer_folder, word_hash + '.pickle')
            if not os.path.exists(word_path):
                self.words.remove(word)
                continue

            if self.pages_reverse_tfidf_rank.get(word, None) is None:
                self.pages_reverse_tfidf_rank[word] = {}

            with open(word_path, 'rb') as f:  # Open word index .pkl file to get link index location
                while True:
                    try:
                        single_page_info = pickle.load(f)  # [doamin, folder_name]
                        self.pages_reverse_tfidf_rank[word][single_page_info[2]] = single_page_info[0]*single_page_info[1]
                        self.all_pages.add(single_page_info[2])

                    except:
                        break

        # pprint.pprint(self.pages_tfidf_rank)

    def get_tf_idf_rank(self):
        '''
        Calculate reversed rank
        '''
        self.get_tf_idf()

        for word in self.words:
            page_reverse_rank = sorted(self.pages_reverse_tfidf_rank[word].items(), key=lambda x: x[1])  # [(url, tfidf)] from low to high
            # pprint.pprint(page_reverse_rank)
            pages_num = len(self.pages_reverse_tfidf_rank[word])
            for reverse_rank in range(1, pages_num + 1):
                self.pages_reverse_tfidf_rank[word][page_reverse_rank[reverse_rank - 1][0]] = reverse_rank / pages_num

        # pprint.pprint(self.pages_tfidf_rank)

    def get_page_rank(self):
        '''
        Calculate reversed rank
        '''
        with open('../data/index/pgrk.pickle', 'rb') as f:
            page_rank = pickle.load(f)

        for url in self.all_pages:
            rank = page_rank.get(url, None)
            if rank is None:
                continue

            self.pages_reverse_page_rank[url] = rank

        pages_num = len(self.pages_reverse_page_rank.keys())
        page_reverse_rank = sorted(self.pages_reverse_page_rank.items(), key=lambda x: x[1], reverse=True)
        for reverse_rank in range(1, pages_num + 1):
            self.pages_reverse_page_rank[page_reverse_rank[reverse_rank - 1][0]] = reverse_rank / pages_num

        # pprint.pprint(self.pages_reverse_page_rank)

    # def get_tf_idf_of_query(self):
    #     with open('../data/index/corpus.json', 'r') as f:
    #         corpus = json.load(f)
    #
    #     with open('../data/index/global_info.json', 'r') as f:
    #         page_number = json.load(f)
    #
    #     for word in self.query.split(' '):
    #         num_of_corpus_contain_word = corpus.get(word, None)
    #         if num_of_corpus_contain_word is None:  # It means the word never appears.
    #             idf = 0
    #         else:
    #             idf = math.log(page_number / (num_of_corpus_contain_word + 1))
    #
    #         tf = 1 / len(self.words)
    #
    #     for index in range(len(self.words)):
    #         self.query_tf_idf[self.words[index]] = tf * idf
    #
    #     print(self.query_tf_idf)

    def rank(self, tf_idf_weight=1.0, page_rank_weight=0.0):
        self.get_tf_idf_rank()
        self.get_page_rank()

        temp_rank = {}
        # print(self.pages_reverse_page_rank['https://jgarcia.ics.uci.edu/?page_id=65'])
        # print(self.pages_reverse_tfidf_rank['uci']['https://jgarcia.ics.uci.edu/?page_id=65'])
        # print(self.pages_reverse_page_rank['https://jgarcia.ics.uci.edu/?page_id=65'] * page_rank_weight +
        #       self.pages_reverse_tfidf_rank['uci']['https://jgarcia.ics.uci.edu/?page_id=65'] * tf_idf_weight)
        # print('======')
        # print(self.pages_reverse_page_rank['https://www.ics.uci.edu/~cs237/'])
        # print(self.pages_reverse_tfidf_rank['uci']['https://www.ics.uci.edu/~cs237/'])
        # print(self.pages_reverse_page_rank['https://www.ics.uci.edu/~cs237/'] * page_rank_weight +
        #       self.pages_reverse_tfidf_rank['uci']['https://www.ics.uci.edu/~cs237/'] * tf_idf_weight)

        for url in self.all_pages:
            url_relative_rank = 0

            relative_rank = self.pages_reverse_page_rank.get(url, None)
            if relative_rank is not None:
                url_relative_rank += relative_rank * page_rank_weight

            for word in self.words:
                relative_rank = self.pages_reverse_tfidf_rank[word].get(url, None)
                if relative_rank is not None:
                    url_relative_rank += relative_rank * tf_idf_weight

            temp_rank[url] = url_relative_rank

        page_rank = sorted(temp_rank.items(), key=lambda x:x[1], reverse=True)  # From high to low
        for value in page_rank:
            self.final_rank.append(value[0])

        return self.final_rank


if __name__ == '__main__':
    # start_time = time.time()
    # query = sys.argv[1]
    queries = [
        "computer vision",
        "academic student employee",
        "natural language",
        "software testing",
        "international student",
        "graphic",
        "information"
    ]
    q1 = open("./results.json", 'w')
    d = {}
    for q in queries:
        start_time = time.time()
        indexer = Indexer(indexer_folder='../data/index/indexer/', query=q)
        d[q] = {}
        d[q]['0.95'] = indexer.rank(tf_idf_weight=0.95, page_rank_weight=0.05)[:10]
        # indexer.rank()
        time1 = time.time()
        print("time1:", str(time1 - start_time))
        time2 = time.time()
        indexer = Indexer(indexer_folder='../data/index/indexer/', query=q)
        d[q]['0.85'] = indexer.rank(tf_idf_weight=0.85, page_rank_weight=0.15)[:10]
        print("time2:", str(time.time() - time2))
    json.dump(d, q1, indent=4)