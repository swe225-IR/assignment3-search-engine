import json
import os
import pickle


class Database:
    def __init__(self, word_save_path, tfidf_path):
        """
        data structure:
            {sub-folder(): [link_file_name1, ...]}
            eg. {xxx_xxx_xxx_edu: [xxxxxxxx.pickle, xxxxxx.pickle, ...]}
        """
        self.word_save_path = word_save_path  # '../data/index/words_index/'
        self.tfidf_path = tfidf_path  # '../data/index/tfidf/'
        self.temp_word_save_path = word_save_path  # # '../data/index/words_index/a/b/c/'
        self.word_info = dict()

    def save(self, pkl_file, tfidf_sub_folder, word_info):
        """
        :param pkl_file: xxxxx.pickle
        :param tfidf_sub_folder: xxx_xxx_xxx_edu
        :param word_info: {word: [tf, idf]}
        """
        for word in word_info:
            if not word.isalnum():
                continue

            self.temp_word_save_path = self.word_save_path
            self.valid_folder(word)
            if not os.path.exists(self.temp_word_save_path + '/' + word + '.pickle'):
                with open(self.temp_word_save_path + '/' + word + '.pickle', 'wb') as fi:
                    # new_info_dic = dict()
                    # new_info_dic[tfidf_sub_folder] = [pkl_file]
                    # pickle.dump(new_info_dic, f)
                    pickle.dump([tfidf_sub_folder, pkl_file], fi)
                    continue

            # with open(self.temp_word_save_path + '/' + word + '.pickle', 'rb') as f:
            #     saved_info_dic = pickle.load(f)
            #     pkl_file_list = saved_info_dic.get(tfidf_sub_folder, None)
            #     if pkl_file_list is None:
            #         saved_info_dic[tfidf_sub_folder] = [pkl_file]
            #     else:
            #         saved_info_dic[tfidf_sub_folder] = pkl_file_list.append(pkl_file)

            # with open(self.temp_word_save_path + '/' + word + '.pickle', 'wb') as f:
            #     pickle.dump(saved_info_dic, f)
            with open(self.temp_word_save_path + '/' + word + '.pickle', 'ab') as fi:
                pickle.dump([tfidf_sub_folder, pkl_file], fi)

    def valid_folder(self, word):
        """
        Creat tree folder for word
        """
        if not os.path.exists(self.temp_word_save_path):
            os.mkdir(self.temp_word_save_path)

        for index in range(0, 3):
            if index > len(word):
                break

            self.temp_word_save_path = os.path.join(self.temp_word_save_path, word[index: index + 1])
            if not os.path.exists(self.temp_word_save_path):
                os.mkdir(self.temp_word_save_path)

    def reset(self):
        self.temp_word_save_path = self.word_save_path  # '../data/index/words_index/'
        self.word_info = dict()


def read_pkl(path):
    """
    :return: word information. eg. {word: [tf, idf]}
    """
    with open(path, 'rb') as f:
        page_info = pickle.load(f)
        for k in page_info:
            url = k
            break
        return page_info[url]


if __name__ == '__main__':
    with open('../data/index/global_info.json', 'r') as f:
        page_number = json.load(f)

    word_save_folder = '../data/index/words_index/'
    tfidf_folder = '../data/index/tfidf/'

    print('Start building index...')
    page_progress = 0
    index_builder = Database(word_save_folder, tfidf_folder)
    for dir_, _, files in os.walk(tfidf_folder):
        for file_name in files:  # file_name : xxx.pickle
            if file_name is None:
                continue

            rel_dir = os.path.relpath(dir_, tfidf_folder)  # rel_dir: xxx_xxx_xxx_edu
            rel_file = os.path.join(rel_dir, file_name)  # xxx_xxx_xxx_edu/xxx.pickle

            # todo
            # print(rel_dir)
            # print(file_name[:-7])
            word_info = read_pkl(tfidf_folder + rel_file)
            index_builder.save(tfidf_sub_folder=rel_dir, pkl_file=file_name[:-7], word_info=word_info)
            index_builder.reset()

            page_progress += 1
            if page_progress % 100 == 0:
                print('Progress: ' + str(int((page_progress / page_number) * 100)) + '%')


