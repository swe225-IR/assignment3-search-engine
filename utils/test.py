import indexer
import time
import sys
import pickle
import hashlib
import os
import pprint


class TestIndexer(indexer.Indexer):
    def __init__(self, indexer_folder: str):
        super(TestIndexer, self).__init__(indexer_folder)
        self.word_operation_time = 0  # Time in word operation (Include stemming, hash)
        self.pkl_operation_time = 0  # Time in ".pickle" file operation (Include open, load)

    def get_tf_idf(self, query: str):
        answer_pages_info = {}

        start_time = time.time()
        words = super().standardize_(query)
        self.word_operation_time += time.time() - start_time

        for word in words:
            start_time = time.time()
            word_hash = hashlib.md5(word.encode()).hexdigest()
            word_path = os.path.join(self.indexer_folder, word_hash + '.pickle')
            self.word_operation_time += time.time() - start_time

            if not os.path.exists(word_path):
                continue

            if answer_pages_info.get(word, None) is None:
                answer_pages_info[word] = []

            start_time = time.time()
            with open(word_path, 'rb') as f:  # Open word index .pkl file to get link index location
                while True:
                    try:
                        single_page_info = pickle.load(f)  # [doamin, folder_name]
                        answer_pages_info[word].append(single_page_info)
                    except:
                        break
            self.pkl_operation_time += time.time() - start_time
        # pprint.pprint(answer_pages_info)


def time_printer(title:str, level:int, numerator, denominator):  # numerator / denominator
    whitespace = '  ' * 2 * level
    content = whitespace + '|--' + title + ': ' + format(numerator, '.4f') + ' ' + format((numerator / denominator) * 100,
                                                                                      '.4f') + '%'
    if (numerator / denominator) * 100 > 30:
        content = '\033[31m' + content + '\033[0m'
    print(content)


if __name__ == '__main__':
    start_time = time.time()
    query = sys.argv[1]
    indexer = TestIndexer(indexer_folder='../data/index/indexer/')
    indexer.get_tf_idf(query)
    total_time = time.time() - start_time
    print('Time in total: ' + format(total_time, '.4f'))
    time_printer('word_operation_time', 1, indexer.word_operation_time, total_time)
    time_printer('pkl_operation_time', 1, indexer.pkl_operation_time, total_time)
