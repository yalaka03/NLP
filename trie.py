import pprint
class Trie:
    def __init__(self,n=2):
        self.root = {}
        self.depth = n
        self.frequency={}
        self.tot_count = 0
        self.vocab = {}

    def insert(self, ngram):
        for x in ngram:
            if x in self.vocab:
                self.vocab[x] += 1
            else:
                self.vocab[x] = 1
        self.tot_count += 1
        node = self.root
        for token in ngram:
            if token not in node:
                node[token] = {"#count": 0}
            node = node[token]
            node["#count"] += 1

    def probability(self, ngram):
        node = self.root
        prev_count = len(self.vocab)
        prob = 0
        for token in ngram:
            if token not in node:
                return 0
            node = node[token]
            prob = node["#count"] / prev_count
            prev_count = node["#count"]
        return prob
    

    def count(self, ngram):
        node = self.root
        for token in ngram:
            if token not in node:
                return 0
            node = node[token]
        return node["#count"]

    def freq_freq(self,root=None):
        #calculate frequency of frequency
        if root is None:
            root = self.root
        if len(root) == 1:
            if root["#count"] in self.frequency:
                self.frequency[root["#count"]] += 1
            else:
                self.frequency[root["#count"]] = 1
        for key in root:
            if key != "#count":
                self.freq_freq(root[key])
        #sort the dict by keys
    
    def return_node(self,ngram):
        node = self.root
        for token in ngram:
            if token not in node:
                return None
            node = node[token]
        return node
    
    def print(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.frequency)
        pp.pprint(self.root)
        
    
    


        

