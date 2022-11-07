from wiktionaryparser import WiktionaryParser
import json

parser = WiktionaryParser()
word = parser.fetch('test')
for k, v in word[0].items():
    print(k + ':')
    print(v)
    print()
wordst = json.dumps(word, indent=4)
print(wordst)
word = json.loads(wordst)
