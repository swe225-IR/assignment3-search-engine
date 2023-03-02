import pickle
import os


def query(domain, url_hash, word, folder='../data/index/tfidf/'):
    path = os.path.join(folder, domain, url_hash)
    print('folder: ' + path)
    with open(os.path.join(path, 'URL_info.pickle'), 'rb') as f:
        url = pickle.load(f)
    print(url)

    for index in (range(3) if len(word) >= 3 else range(len(word) - 1)):
        path = os.path.join(path, word[index])

    if not os.path.exists(path):
        return

    with open(os.path.join(path, word+'.pickle'), 'rb') as f:
        tf_idf = pickle.load(f)
    print(tf_idf)

if __name__ == '__main__':
    query('motifmap-rna_ics_uci_edu', '920dbd560e078232b99223bbc3435c0e7fe6d7660e0deb17d0b2bf17730b892c', 'factor')