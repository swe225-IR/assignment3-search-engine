import indexer
import time
import sys
import pickle


from nltk.stem.snowball import SnowballStemmer


class TestIndexer(indexer.Indexer):
    def __init__(self, tfidf_path: str, words_index_path: str):
        super(TestIndexer, self).__init__(tfidf_path, words_index_path)
        self.stemmer_time = 0  # Time in stemming
        self.word_index_time = 0  # Time in word indexer
        self.link_index_time = 0  # Time in link indexer
        self.link_pkl_load_time = 0  # Time in load pickle file
        self.link_open_file_time = 0  # Time in opening pickle file

    def word_indexer(self, word: str):
        start_time = time.time()
        word_path = super().word_indexer(word)
        self.word_index_time += time.time() - start_time
        return word_path

    def link_indexer(self, locations_info: list, word: str):
        link_index_start_time = time.time()

        link_info_s = {}
        urls = []
        tf_idf_scores = []
        tf_idf_paths = super().path_transformer(locations_info, word)
        url_paths = super().get_urls(locations_info)

        open_time = time.time()
        for url_path in url_paths:
            start_time = time.time()
            f = open(url_path, 'rb')
            self.link_open_file_time += time.time() - start_time

            start_time = time.time()
            urls.append(pickle.load(f))
            self.link_pkl_load_time += time.time() - start_time
            f.close()

        for tf_idf_path in tf_idf_paths:
            start_time = time.time()
            f = open(tf_idf_path, 'rb')
            self.link_open_file_time += time.time() - start_time

            start_time = time.time()
            tf_idf_scores.append(pickle.load(f))
            self.link_pkl_load_time += time.time() - start_time
            f.close()

        for index in range(len(urls)):
            link_info_s[urls[index]]  = tf_idf_scores[index]  # {url: [tf, idf]}

        self.link_index_time += time.time() - link_index_start_time
        return link_info_s

    def standardize_(self, query: str, stemmer=SnowballStemmer(language="english")):
        start_time = time.time()
        word_list = super().standardize_(query, stemmer)
        self.stemmer_time += time.time() - start_time
        return word_list


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
    indexer = TestIndexer(tfidf_path='../data/index/tfidf/', words_index_path='../data/index/words_index/')
    indexer.get_tf_idf(query)
    total_time = time.time() - start_time
    print('Time in total: ' + format(total_time, '.4f'))
    time_printer('Stemming', 1, indexer.stemmer_time, total_time)
    time_printer('Word index', 1, indexer.word_index_time, total_time)
    time_printer('Link index', 1, indexer.link_index_time, total_time)
    time_printer('Pkl open time', 2, indexer.link_open_file_time, indexer.link_index_time)
    time_printer('Pkl load time', 2, indexer.link_pkl_load_time, indexer.link_index_time)
