import json
if __name__ == '__main__':
    f= open("E:\\inv_index.json",'r')
    a = json.load(f)
    print(a['world'])