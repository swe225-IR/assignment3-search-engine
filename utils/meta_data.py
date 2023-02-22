import json
from html.parser import HTMLParser
from html.entities import name2codepoint
from lxml import etree


# <script>...</script> and <style>...</style>
# normal: 1, b: 2, h4: 3, h3: 4, h2: 5, h1: 6, title: 7

class MyHTMLParser(HTMLParser):

    def __init__(self):

        super().__init__()
        self.stack = []

    def handle_starttag(self, tag, attrs):
        print("Start tag:", tag)
        self.stack.append(tag)
        # for attr in attrs:
        #     print("     attr:", attr)

    def handle_endtag(self, tag):
        self.stack.pop()
        # print("End tag  :", tag)

    def handle_data(self, data):
        # print("Data     : t/", self.stack[-1], data)
        pass

    def handle_entityref(self, name):
        c = chr(name2codepoint[name])
        print("Named ent:", c)

    def handle_charref(self, name):
        if name.startswith('x'):
            c = chr(int(name[1:], 16))
        else:
            c = chr(int(name))
        print("Num ent  :", c)

    def handle_decl(self, data):
        print("Decl     :", data)


if __name__ == '__main__':
    f = open("../da5aff1b5ca2bad6609f97f11c91fef3a503ded6d9d0f14592793c9391b92fd9.json", 'r')
    a = json.load(f)
    s = a['content']
    root = etree.HTML(s)
    for bad in root.xpath("//script"):
        bad.getparent().remove(bad)
    for bad in root.xpath("//style"):
        bad.getparent().remove(bad)

    parser = MyHTMLParser()
    parser.feed(etree.tostring(root).decode('utf-8'))
