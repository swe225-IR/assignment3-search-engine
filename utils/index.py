import json
import os
import sys
import pickle as pkl
from urllib.parse import urlparse

import nltk
from bs4 import BeautifulSoup
from lxml import etree
from nltk import WordNetLemmatizer, word_tokenize
from nltk.stem.porter import PorterStemmer
from simhash import Simhash


TAGS_ABANDON = ['CC', 'DT', 'FW', 'IN', 'LS', 'PDT', 'PRP', 'PRP$', 'RP', 'SYM', 'TO', 'PP']
TAGS_VERB = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VP']
TAGS_ADJ = ['JJ', 'JJR', 'JJS', 'ADJP', 'ADVP']
TAGS_NOUN = ['NN', 'NNS', 'NNP', 'NNPS', 'NP']
TAGS_ADV = ['RB', 'RBR', 'RBS']
ADV_OTHERS = ['CD', 'EX', 'MD', 'UH', 'WDT', 'WP', 'WP$', 'WRB', 'SBAR', 'PRT', 'INTJ', 'PNP', '-SBJ', '-OBJ']
STOP_WORDS = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't",
              'as',
              'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't",
              'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down',
              'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't",
              'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself',
              'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's",
              'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of',
              'off',
              'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same',
              "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that',
              "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they',
              "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until',
              'up',
              'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's",
              'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with',
              "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself',
              'yourselves']
WORD_ABBREVIATION = ['re', 've', 'll', 'ld', 'won', 'could', 'might', 'isn', 'aren', 'couldn', 'hasn', 'haven', 'wasn',
                     'weren']


class Page:
    def __init__(self, path):
        self.raw_content: str = ''
        self.url: str = ''
        self.encoding: str = ''
        self.content = ''
        self.path: str = path
        self.forward_index = {}
        self.inverted_index = {}

    def read_json(self):
        f = open(self.path, 'rb')
        page_json = json.load(f)
        self.raw_content = page_json['content']
        ind = page_json['url'].find("#")
        if ind != -1:
            self.url = page_json['url'][:ind]
        else:
            self.url = page_json['url']
        self.encoding = page_json['encoding']

    def extract_word(self):
        if self.raw_content:
            soup = BeautifulSoup(self.raw_content, features='html.parser')
            self.content = soup.get_text()  # todo: h1,h2,h3

    def standardize_words(self):
        hash_values_path = 'hash_values.pkl'
        lemmatizer = WordNetLemmatizer()
        word_list = []
        unstandardized_words = word_tokenize(self.content.lower())
        word_pos_tags = nltk.pos_tag(unstandardized_words)
        for word_pos_tag in word_pos_tags:
            word = word_pos_tag[0].replace(' ', '')

            # Remove stopwords
            if self.stopwords_filter(word) == '':
                continue

            # Special case filter
            if self.special_case_filter(word) == '':
                continue
            # Get the classification of words and do the initial filter
            wordnet_tag = self.pos_tags_filter(word_pos_tag[1])
            if wordnet_tag == '':
                continue
            elif wordnet_tag == 'add':
                word_list.append(word)
                # current_page_word_num += 1
            else:
                # Lemmatization todo: use lemmatization or stemming?
                standardize_word = lemmatizer.lemmatize(word, wordnet_tag)
                word_list.append(standardize_word)
        flag = self.similarity_comparison(word_list=word_list, f_path=hash_values_path)
        if flag is False:
            for i in range(0, len(word_list)):
                if self.forward_index.__contains__(word_list[i]):
                    self.forward_index[word_list[i]].append(i)
                else:
                    self.forward_index[word_list[i]] = [i]
            for key, value in self.forward_index.items():
                self.inverted_index[key] = {self.url: value}

    def run(self):
        self.read_json()
        self.extract_word()
        self.standardize_words()

    def stopwords_filter(self, word: str) -> str:
        if word in STOP_WORDS:
            return ''
        return word

    def special_case_filter(self, word: str) -> str:
        if len(word) == 1:
            return ''
        elif (len(word) >= 2) and (word in WORD_ABBREVIATION):
            return ''
        return word

    def pos_tags_filter(self, tag: str) -> str:
        """
        Filter the words that belong to the tags we do not need and get wordnet compatible tags
        :param tag: Pos tags
        :return: Wordnet compatible tags
        """
        if tag in TAGS_ADJ:
            return nltk.corpus.wordnet.ADJ
        elif tag in TAGS_VERB:
            return nltk.corpus.wordnet.VERB
        elif tag in TAGS_NOUN:
            return nltk.corpus.wordnet.NOUN
        elif tag in TAGS_ADV:
            return nltk.corpus.wordnet.ADV
        elif tag in ADV_OTHERS:
            return 'add'
        else:
            return ''

    def hamming_distance(self, int_a, int_b):
        x = (int_a ^ int_b) & ((1 << 64) - 1)
        ans = 0
        while x:
            ans += 1
            x &= x - 1
        return ans

    def similarity_comparison(self, word_list: list, f_path: str, hash_threshold=8) -> bool:
        page_hash_value = Simhash(word_list).value
        if os.path.isfile(f_path) is False:
            print(os.curdir)
            f = open(f_path, 'wb')
            pkl.dump([page_hash_value], f)
            f.close()
            return False
        else:
            f = open(f_path, 'rb')
            hash_values = pkl.load(f)
            f.close()
            for hash_value in hash_values:
                if self.hamming_distance(hash_value, page_hash_value) <= hash_threshold:
                    print("[simhash filter] -> " + self.url)
                    return True
            f = open(f_path, 'wb')
            hash_values.append(page_hash_value)
            pkl.dump(hash_values, f)
            f.close()
            return False


if __name__ == '__main__':
    """
    sys.argv[1] -> absolute value
    """
    forward_index = {}
    inverted_index = {}
    dev_path = sys.argv[1]
    domains = os.listdir(dev_path)
    for d in domains:
        files = os.listdir(f'{dev_path}/{d}')
        for fi in files:
            p = Page(f'{dev_path}/{d}/{fi}')
            p.run()
            for k, v in p.inverted_index.items():
                if inverted_index.__contains__(k):
                    for k1, v1 in v.items():
                        inverted_index[k][k1] = v1
                else:
                    inverted_index[k] = {}
                    for k1, v1 in v.items():
                        inverted_index[k][k1] = v1

    inverted_index_fp = open("inv_index.json", 'a')
    json.dump(inverted_index, inverted_index_fp, indent=4)
