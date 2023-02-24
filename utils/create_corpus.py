import json
import os


from collections import Counter


if __name__ == '__main__':
    root_dir = '../METADATA/'
    corpus_path = '../data/index/corpus.json'
    global_info_path = '../data/index/global_info.json'

    page_number = 0  # page number in total
    corpus = Counter()
    for dir_, _, files in os.walk(root_dir):
        for file_name in files:
            rel_dir = os.path.relpath(dir_, root_dir)
            rel_file = os.path.join(rel_dir, file_name)
            page_number += 1
            with open(root_dir + rel_file, 'r') as f:
                page = json.load(f)
            word_per_page = page[[*page][0]]['word_positions']
            for word in word_per_page:  # Calculate the number of pages that contains word.
                corpus[word] += 1

    with open(corpus_path, "w") as outfile:
        json.dump(corpus, outfile, indent=4)

    with open(global_info_path, "w") as outfile:
        json.dump(page_number, outfile, indent=4)

    print('Page_number: ' + str(page_number) + '\n' + 'Done')
