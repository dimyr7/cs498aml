#! env python
import numpy as np
import random as rand
import math

class Theta(object):
    def __init__(self, num_topics, num_words):
        self.num_topics = num_topics
        self.num_words = num_words

        pi_temp = np.random.rand(num_topics)
        self.pi = pi_temp / pi_temp.sum()


        pvec_temp = np.random.rand(num_topics, num_words)
        self.pvec = pvec_temp / pvec_temp.sum(axis=0)
        print self.pvec


    def get_w_ij(self, x, (i,j)):
        ret_sum = math.log(self.pi[j])
        for k in range(self.num_words):
            temp_sum = x[i,k]*math.log(self.pvec[j,k])
            #print("k=" + str(k) + ", temp_sum=" + str(temp_sum) + ", accum_sum=" + str(ret_sum))
            ret_sum += temp_sum
        print("i=" +str(i) + ", j=" + str(j) + ", e^sum=" + str(math.exp(ret_sum)))

        return math.exp(ret_sum)

    def get_w_ij_old(self, x, (i,j)):
        ret_prod = 1.
        for k in range(self.num_words):
            ret_prod *= (self.pvec[j,k] ** x[i,k])
        return ret_prod

    def get_w(self, x):
        w = np.zeros((len(x), self.num_topics))
        for i in range(len(x)):
            for j in range(self.num_topics):
                w[i,j] = (self.get_w_ij(x, (i,j)))
            if(np.all(w[i] == np.zeros(self.num_topics))):
                print("ERRROR, ALL ZEROS, DIVIDE BY 0 PENDING, i=" + str(i))
                exit(1)
            print w[i]
            w[i] = w[i]/ w[i].sum()
        return w




## Reading the data
docword_path = "./docword.nips.txt"
vocab_path   = "./vocab.nips.txt"

docword = open(docword_path, "r")
vocab   = open(vocab_path,   "r")

num_documents = int(docword.readline())
num_words     = int(docword.readline())
num_entries   = int(docword.readline())
num_topics = 30



dictionary = [""]*num_words

for i in range(num_words):
    dictionary[i] = vocab.readline()


data = np.zeros((num_documents, num_words))
for i in range(num_entries):
    doc_info = map(int, docword.readline().split())
    data[doc_info[0] - 1 , doc_info[1] - 1] = doc_info[2]


## Initial conditions

theta = Theta(num_topics, num_words)
for iteration in range(30):
    print("Starting iteration " + str(iteration))
    w = theta.get_w(data)
    for j in range(num_topics):
        theta.pi[j] = w.sum(axis=0)/num_documents
    for j in range(num_topics):
        temp_num = np.zeros(num_topics)
        for i in range(num_documents):
            temp_num += data[i] * w[i,j]
        temp_den = 0.
        for i in range(num_documents):
            temp_den += x[i].sum() * w[i,j]
        theta.pvec[j] = temp_num/temp_den

theta1 = Theta(num_topics, num_words)
for iteration in range(30):
    print("Starting iteration " + str(iteration))
    w = theta1.get_w(data)
    for j in range(num_topics):
        theta1.pi[j] = w.sum(axis=0)/num_documents
    for j in range(num_topics):
        temp_num = np.zeros(num_topics)
        for i in range(num_documents):
            temp_num += data[i] * w[i,j]
        temp_den = 0.
        for i in range(num_documents):
            temp_den += x[i].sum() * w[i,j]
        theta1.pvec[j] = temp_num/temp_den


print (theta.pi - theta1.pi)

print "All done"
