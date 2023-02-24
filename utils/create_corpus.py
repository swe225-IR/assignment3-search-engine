import json
import os
from collections import Counter


if __name__ == '__main__':
    root_dir = 'D:/UCI/1-Q2/CS221/Assignment-3-web-crawler/assignment3-search-engine/METADATA/'
    corpus_path = 'D:/UCI/1-Q2/CS221/Assignment-3-web-crawler/assignment3-search-engine/corpus.json'
    rel_files = []
    corpus = Counter()

    for dir_, _, files in os.walk(root_dir):
        for file_name in files:
            rel_dir = os.path.relpath(dir_, root_dir)
            rel_file = os.path.join(rel_dir, file_name)
            rel_files.append(rel_file)
            with open(root_dir + rel_file, 'r') as f:
                page = json.load(f)
            word_per_page = page[[*page][0]]['word_positions']
            for word in word_per_page:
                corpus[word] += 1

    corpus['total page number'] = len(rel_files)

    with open(corpus_path, "w") as outfile:
        json.dump(corpus, outfile, indent=4)