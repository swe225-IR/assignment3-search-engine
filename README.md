# assignment3-search-engine

## Building Index

1. Step into the project directory

2. Execute the following commands:

```bash
conda create -n a2 python=3.10
conda activate a2
pip install -r requirements.txt
rm -rf *.pkl output_of_index.log ./METADATA/*
nohup python utils/metadata_builder.py your/abs/path/to/DEV/ > output_of_index.log 2>&1 &
```

3. Check metadata of the pages:

```bash
cd ./METADATA
```

## Indexer

### Description

In the indexer file system, a word and its related information are stored as a “.pickle” file and the file
name is the hash value of the word. The file content is as follows. The scores mean the “TF” and
“IDF” scores of the word in a URL.

```text
[TF score, IDF score, URL_1],
[TF score, IDF score, URL_2],
......
```

The process of retrieving information from the indexer is that when we search for a word, the indexer
will calculate the hash value of the word and use the hash value to find the corresponding “.pickle”
directly.

### Summary

The size of the entire indexer is 1.3GB.

| Indexer size | Number of document | Number of word |
|--------------|--------------------|----------------|
| 1,363,148 KB | 25,058             | 1,610,484      |


## Calculate Pagerank

We have already supplied the complete set of crawled pages'
links: [original_link_set.json](data/links/original_link_set.json).
Additionally, from the output_of_index.log, we can determine which [links are duplicated](tmp/duplicate_link.json) and
will be removed.

Next, we extract links from each webpage using the original data (with `doc.xpath("//a")`), which is stored
in [link_out_edges.json](data/links/link_out_edges.json). To run the PageRank algorithm, execute python pagerank.py
within the utils directory. After filtering, the PageRank values for each page can be found
in [pgrk.json](../temp/pgrk.json)

## Detection and elimination of duplicate pages

Hamming distance with a threshold of 8 is deemed suitable for our case, as we have conducted experiments and determined that this threshold yields the best results, see [web crawler](https://github.com/swe225-IR/assignment2-crawler).
```python
def hamming_distance(int_a, int_b):
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
                if hamming_distance(hash_value, page_hash_value) <= hash_threshold:
                    print(f"[SHFilter] -> {self.url}")
                    return True
            f = open(f_path, 'wb')
            hash_values.append(page_hash_value)
            pkl.dump(hash_values, f)
            f.close()
            return False
```

## License
GNU General Public License v3.0