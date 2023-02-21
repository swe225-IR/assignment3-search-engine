import os
import re
import networkx as nx
from typing import List
import json
from urllib.parse import urlparse, ParseResult
import numpy as np

from lxml import etree
from numpy import float32


class Node:
    def __init__(self, url: str, outgoing_url: List[str]):
        self.url = url
        self.outgoing_url = outgoing_url
        self.out_edges_num = len(outgoing_url)

    def out_edges(self):
        return self.out_edges_num

    def out_edges_contains_delete(self, url):
        ind = self.outgoing_url.index(url)
        if ind != -1:
            self.outgoing_url.remove(url)
            self.out_edges_num -= 1


def get_all_links(args):
    """
    sys.argv[1] -> absolute value
    """
    link_set = set()
    dev_path = args[1]
    domains = os.listdir(dev_path)
    for d in domains:
        files = os.listdir(f'{dev_path}/{d}')
        for fi in files:
            f = open(f'{dev_path}/{d}/{fi}')
            a = json.load(f)
            url = a['url']
            ind = url.find("#")
            if ind != -1:
                link_set.add(url[:ind])
            else:
                link_set.add(url)
    fff = open("../original_link_set.json", 'a')
    json.dump(list(link_set), fff)


def get_links(args):
    f_l = open("../original_link_set.json", 'r')
    l_s = set(json.load(f_l))
    dev_path = args[1]
    domains = os.listdir(dev_path)
    nodes = {}
    for d in domains:
        files = os.listdir(f'{dev_path}/{d}')
        for fi in files:
            f = open(f'{dev_path}/{d}/{fi}')
            a = json.load(f)
            content = a['content']
            cur_url = a['url']
            if cur_url not in l_s:
                continue
            out_links = outgoing_urls(cur_url, content)
            out_links = [x for x in out_links if x in l_s]
            n = Node(cur_url, out_links)
            nodes[n.url] = n.outgoing_url
    ffff = open("link_out_edges.json", 'a')
    json.dump(nodes, ffff, indent=4)


def outgoing_urls(url, content):
    if not content:
        return []
    root = etree.HTML(content.encode('utf-8'))
    if not root:
        return []
    a_nodes = root.xpath("//a")
    if not a_nodes:
        return []
    cur_lk_p = urlparse(url)
    links = [x.get('href') for x in a_nodes if x.get('href')]
    for i in range(0, len(links)):
        if is_url_defense(links[i]):
            pass
        else:
            parsed = urlparse(links[i])
            url_processed = handle_urls(links[i], parsed)
            if parsed.scheme == '':
                if parsed.netloc == '':  # href = "/xxxxx"
                    links[i] = f"{cur_lk_p.scheme}://{cur_lk_p.netloc}{url_processed}"
                else:  # href = "//www.xxx.xxx/xxxxxx"
                    links[i] = f"{cur_lk_p.scheme}:{url_processed}"
            else:
                links[i] = url_processed
    return links


def handle_urls(origin_url: str, parsed: ParseResult) -> str:
    """
    processing URL string
    :param origin_url: URL of this web page
    :param parsed: urlparse(origin_url)
    :return: sorted string
    """
    if parsed.query == '' and parsed.params == '':
        return parsed.geturl().split("#")[0]
    elif parsed.query == '' and parsed.params != '':
        params_str = handle_params_or_query(parsed.params, ";")
        p_id = origin_url.find(";")
        return f'{origin_url[:p_id]};{params_str}'
    elif parsed.query != '' and parsed.params == '':
        query_str = handle_params_or_query(parsed.query, "&")
        q_id = origin_url.find("?")
        return f'{origin_url[:q_id]}?{query_str}'
    else:
        query_str = handle_params_or_query(parsed.query, "&")
        params_str = handle_params_or_query(parsed.params, ";")
        p_id = origin_url.find(";")
        return f'{origin_url[:p_id]};{params_str}?{query_str}'


def handle_params_or_query(params_or_query_str: str, separator: str) -> str:
    """
    to handle the situation that two params or query strings are exactly the same except for the order
    :param params_or_query_str: params or params1=value1;params2=value2 or query1=value1&query2=value2
    :param separator: ; or &
    :return: sorted list
    """
    pair = params_or_query_str.split(separator)
    result_list = list()
    for p in pair:
        pair_list = p.split("=")
        if len(pair_list) == 2:
            result_list.append((pair_list[0], pair_list[1]))
        else:
            result_list.append((pair_list[0], ''))
    result_list.sort()
    url_partial = ''
    for r in result_list:
        url_partial += f'{r[0]}={r[1]}{separator}'
    return url_partial[:-1]  # discarding the separator at the end


def get_page_rank():
    nodes_json = json.load(open("../link_out_edges.json", 'r'))
    except_json = json.load(open('../duplicate_link.json', 'r'))
    G = nx.DiGraph()
    for k, v in nodes_json.items():
        if k not in except_json:
            v_p = [x for x in v if x not in except_json]
            for v1 in v_p:
                G.add_edge(k, v1)

    pgrk = nx.pagerank(G)
    pgrk_ = sorted(pgrk.items(), key=lambda x: x[1], reverse=True)
    fp1 = open("pgrk.json", 'a')
    json.dump(pgrk_, fp1, indent=4)
    # distinct_links.add(k)
    # for v_ in v_p:
    #     distinct_links.add(v_)
    # nodes_list.append(Node(k, list(v_p)))
    # distinct_links = list(distinct_links)
    # nodes_map = {}
    # for d in range(0, len(distinct_links)):
    #     nodes_map[distinct_links[d]] = d
    #
    # M = np.zeros((len(distinct_links), len(distinct_links)), dtype=float32)
    # for j in range(0, len(nodes_list)):
    #     out_ = nodes_list[j].outgoing_url
    #     if len(out_) == 0:
    #         continue
    #     for o in out_:
    #         M[nodes_map[o]][j] = 1.0 / len(out_)
    #
    # d = 0.85
    # I = [1.0] * len(distinct_links)
    # r = np.linalg.inv(np.diag(I) - d * M) * (1 - d) / len(distinct_links) * np.ones((len(distinct_links), 1), dtype=float32)
    # print(r)
    # print(np.nonzero(M))
    # print(M[0, 52365])


def is_url_defense(url: str) -> bool:
    return True if re.compile(r'https://urldefense(?:\.proofpoint)?\.com/(v[0-9])/').search(url) else False


if __name__ == '__main__':
    get_page_rank()
