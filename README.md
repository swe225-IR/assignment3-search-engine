# assignment3-search-engine

## Building Index

1. Step into the project directory

2. Execute the following commands:

```shell
conda create -n a2 python=3.10
conda activate a2
pip install -r requirements.txt
rm -rf *.pkl output_of_index.log ./METADATA/*
nohup python utils/index.py your/abs/path/to/DEV/ > output_of_index.log 2>&1 &
```

3. Check metadata of the pages:

```shell
cd ./METADATA
```

### Calculate Pagerank

We have already provided the total link set of crawled
pages: [original_link_set.json](data/links/original_link_set.json).
Also, from output_of_index.log we can get which [links are duplicated](tmp/duplicate_link.json) (will be
deleted).
Then we extract links from every web pages using the original data (using `doc.xpath("//a")`), which is stored
in [link_out_edges.json](data/links/link_out_edges.json).
Run `python pagerank.py` under the directory of `utils`.
And we can get the value of each page (after being filtered): [pgrk.json](utils/pgrk.json)
