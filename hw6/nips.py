#! env python
import numpy as np
import random as rand

class Theta(object):
    def __init__(self, num_topics, num_words):
        self.num_topics = num_topics
        self.num_words = num_words

        pi_temp = np.random.rand(num_topics)
        self.pi = pi_temp / pi_temp.sum()


        pvec_temp = np.random.rand(num_topics, num_words)
        self.pvec = pvec_temp / pvec_temp.sum(axis=0)



    def get_z(self, x):
        print "get_z"
        z = np.zeros((len(x), self.num_topics))
        for i in range(len(x)):
            print("doc:" + str(i) + "/" + str(len(x)))
            for j in range(self.num_topics):
                z[i,j] = np.log(self.pi[j])
                for k in range(self.num_words):
                    z[i,j] += (x[i,k] * np.log(self.pvec[j,k]))
        return z


    def get_d(self, z):
        print "get_d"
        d = np.zeros(z.shape[0])
        for i in range(len(d)):
            d[i] = np.amax(z[i])
        return d

    def get_w(self, x):
        print "get_w"
        z = self.get_z(x)
        d = self.get_d(z)
        w = np.zeros((len(x), self.num_topics))
        for i in range(len(x)):
            for j in range(self.num_topics):
                w[i, j] = np.exp(z[i,j] - d[i])
        for i in range(len(x)):
            w[i,:] = w[i,:]/np.sum(w[i])
        return w
# =====




## Reading the data
docword_path = "./short.nips.txt"
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
        theta.pi[j] = np.sum(w[:,j])/num_documents
    for j in range(num_topics):
        temp_num = np.zeros(num_words)
        for i in range(num_documents):
            temp_num += data[i] * w[i,j]
        temp_den = 0.
        for i in range(num_documents):
            temp_den += data[i].sum() * w[i,j]
        theta.pvec[j] = temp_num/temp_den


print "All done"
