from tokenizer_util import tokenize
from trie import Trie
import pickle as pkl
import random
import numpy as np
from scipy import stats
import pprint

class NGramModel:
    def __init__(self, corpus_path, n=2, interpolation=False, good_turing=False):
        self.corpus_path = corpus_path
        self.n = n
        self.interpolation = interpolation
        self.good_turing = good_turing
        self.Trie = Trie(n)
    
    def read_file(self):
        with open(self.corpus_path, 'r') as file:
            self.corpus = file.read()
        self.corpus = tokenize(self.corpus)


        return self.corpus

    def setup(self):
        self.corpus = self.read_file()
        test_size = 1000
        # test_size = 1
        self.test = random.sample(self.corpus, test_size)
        self.train = [x for x in self.corpus if x not in self.test]

    def train_model(self):
        for sentence in self.train:
            sentence.append("</s>")
            for i in range(len(sentence)):
                if i < self.n - 1:
                    ngram = sentence[:i+1]
                    while len(ngram) < self.n:
                        ngram.insert(0, "<s>")
                    self.Trie.insert(ngram)
                else:
                    ngram = sentence[i-self.n+1:i+1]
                    self.Trie.insert(ngram)
            if self.interpolation:
                for i in range(len(sentence)-self.n+1, len(sentence)):
                    ngram = sentence[i:]
                    self.Trie.insert(ngram)

        if self.interpolation:
            self.train_interpolation()

        if self.good_turing:
            self.train_good_turing()
        
    def save(self, file_path):
        pkl.dum(self, open(file_path, 'wb'))
    
    def load(self, file_path):
        self = pkl.load(open(file_path, 'rb'))
            
        
    # def perplexity(sentence):
    # <calculate the perplexity of the sentence>
    # def generate():
    # <generate a sentence>
    # <use the probabilities calculated in train() to generate the next word>
    # def evaluate():
    # <evaluate the model>
    # <calculate the average perplexity of the train sentence>
    # <calculate the average perplexity of the test sentence>
    def smoothing(self, r):
        #return exp(a * log(r) + b)
        return np.exp(self.a * np.log(r) + self.b)

    def train_good_turing(self):
        #calculate frequency of frequency
        self.Trie.freq_freq()
        N_r = self.Trie.frequency
        #calculate Z_r
        Z_r = {}
        keys = list(N_r.keys())
        keys.sort()
        for i in range(len(keys) - 1):
            if i == 0:
                Z_r[keys[i]] =  N_r[keys[i]] / (1/2 * (keys[i + 1] - 0))
            elif i == len(keys) - 1:
                Z_r[keys[i]] =  N_r[keys[i]] / ( (keys[i] - keys[i - 1]))
            else : 
                Z_r[keys[i]] =  N_r[keys[i]] /(1/2 * (keys[i + 1] - keys[i-1])) 
        result = stats.linregress(np.log(list(N_r.keys())), np.log(list(N_r.values())))
        self.a = result.slope
        self.b = result.intercept
        # print(self.Trie.frequency)
        # print(self.a, self.b)


    
    def train_interpolation(self):
        self.lambdas = [0] * self.n
        for sentence in self.train:
            for i in range(len(sentence)):
                if i < self.n - 1:
                    ngram = sentence[:i + 1]
                    while len(ngram) < self.n:
                        ngram.insert(0, "<s>")
                    
                else:
                    ngram = sentence[i - self.n + 1:i + 1]
                index = 0
                maximum = -100000000
                for j in range(self.n):
                    if j == self.n - 1:
                        if self.Trie.tot_count-1 == 0:
                            value = 0
                        else:
                            value = (self.Trie.count(ngram[j:])-1)/(self.Trie.tot_count-1)
                    else:
                        if self.Trie.count(ngram[j:-1])-1 == 0:
                            value = 0
                        else:
                            value = (self.Trie.count(ngram[j:])-1)/(self.Trie.count(ngram[j:-1])-1)
                    if value > maximum:
                        maximum = value
                        index = j
                self.lambdas[self.n-1-index] += self.Trie.count(ngram)
        
        # Normalize lambdas
        total = sum(self.lambdas)
        self.lambdas = [l / total for l in self.lambdas]
        # print(self.lambdas)
    
    def interpolated_probability(self, ngram):
        prob = 0
        for i in range(self.n):
            prob+=self.lambdas[self.n-1-i]*self.Trie.probability(ngram[i:])
        # print(prob)
        return prob
    
    def smoothed_count(self, ngram):
        if(self.Trie.count(ngram) == 0):
            return self.smoothing(1)
        else :
            r = self.Trie.count(ngram)
            return (r+1)*(self.smoothing(r+1)/self.smoothing(r))
    
    def smoothing_prob(self, ngram):
        numerator = self.smoothed_count(ngram)
        denominator = 0
        node = self.Trie.return_node(ngram[:-1])
        
        if node is not None:
            for key in node:
                if(key != "#count"):
                    r = node[key]["#count"]
                    denominator += (r+1)*(self.smoothing(r+1)/self.smoothing(r))
            denominator += (len(self.Trie.vocab) - len(node))*(self.smoothing(1))
        else:
            denominator += (len(self.Trie.vocab))*(self.smoothing(1))
        # print(numerator/ denominator)
        return numerator/denominator

    
    def probability(self, sentence):
        for i in range(self.n-1):
            sentence.insert(0, "<s>")
        if(sentence[-1] != "</s>"):
            sentence.append("</s>")
        if self.interpolation:
            prob = 1
            for i in range(self.n-1, len(sentence)):
                ngram = sentence[i-self.n+1:i+1]
                prob*=self.interpolated_probability(ngram)
            return prob   
        elif self.good_turing:
            prob = 1
            for i in range(self.n-1, len(sentence)):
                ngram = sentence[i-self.n+1:i+1]
                prob*=self.smoothing_prob(ngram)
            return prob
        else:
            prob = 1
            for i in range(self.n-1, len(sentence)):
                ngram = sentence[i-self.n+1:i+1]
                prob*=self.Trie.probability(ngram)
            return prob
    
    def perplexity(self, sentence):
        prob = self.probability(sentence)
        if(prob == 0):
            return 0
        return 1/prob**(1/len(sentence))
    
    def print_perplexity(self):
        perplexity = 0
        for i in range(len(self.train)):
            value = self.perplexity(self.train[i])
            perplexity+=value
            print("senetce_",i,"    ",value)
        print("Train Perplexity: ",perplexity/len(self.train))
    
    def generate(self,sentence,top_k):
        while(len(sentence) < self.n-1):
            sentence.insert(0, "<s>")
        sentence = sentence[-self.n+1:]
        sentence.append("</s>")
        words = {}
        for key in self.Trie.vocab:
            sentence[-1] = key
            if self.interpolation:
                words[key] = self.interpolated_probability(sentence)
            elif self.good_turing:
                words[key] = self.smoothing_prob(sentence)
                # print(words[key])
            else:
                words[key] = self.Trie.probability(sentence)
        words = sorted(words.items(), key=lambda x: x[1], reverse=True)
        return words[:top_k]




# model = NGramModel("Ulysses.txt", 3, good_turing=True)
# model.setup()
# model.train_model()
# model.print_perplexity()


        


            
