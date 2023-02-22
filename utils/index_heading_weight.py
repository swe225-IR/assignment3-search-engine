import json
# from html.parser import HTMLParser
from html.entities import name2codepoint
from lxml import etree
from lxml.html import HTMLParser
from lxml import html

except_tags = {}


class MyHTMLParser(HTMLParser):

    def __init__(self):

        super().__init__()
        self.stack = []

    def handle_starttag(self, tag, attrs):
        # print("Start tag:", tag)
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
    html1 = ""

    root = html.fromstring(html1)
    # print(root.xpath("//h2"))
    print(root.text_content())
