#! env python
import numpy as np
import sklearn.cluster
import random as rand
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

epsilon  = 10.**-3
SUBSET_DOC = 500
SUBSET_WORDS = 100
top_x_words = 10
num_topics = 30
num_iterations = 10


class TopicTheta(object):
    def __init__(self, num_topics, x):
        self.num_topics = num_topics
        self.num_words = x.shape[1]

        kmeans = sklearn.cluster.KMeans(n_clusters = num_topics).fit(x)
        self.pi = np.zeros(num_topics)
        for j in range(num_topics):
            self.pi[j] = (kmeans.labels_ == j).sum()/float(x.shape[0])
            if(self.pi[j] == 0):
                self.pi[j] = epsilon


        self.pvec = np.zeros((num_topics, num_words))
        total_word_count = x.sum(axis=0)
        for k in np.where(total_word_count == 0)[0]:
            total_word_count[k] = 1
        for j in range(num_topics):
            self.pvec[j] = np.divide(x[kmeans.labels_ == j].sum(axis=0), total_word_count)
            for k in np.where(self.pvec[j] == 0)[0]:
                self.pvec[j,k] = 1./num_topics


    def get_z(self, x):
        print "get_z"
        z = np.zeros((x.shape[0], self.num_topics))
        for i in range(x.shape[0]):
            if(i%50 == 0):
                print("doc: " + str(i) + "/" + str(x.shape[0]))
            for j in range(self.num_topics):
                z[i,j] = np.log(self.pi[j])
                for k in range(self.num_words):
                    z[i,j] += (x[i,k] * np.log(self.pvec[j,k]))
        return z


    def get_d(self, z):
        print "get_d"
        d = np.zeros(z.shape[0])
        for i in range(z.shape[0]):
            d[i] = np.amax(z[i])
        return d

    def get_w(self, x):
        print "get_w"
        z = self.get_z(x)
        d = self.get_d(z)
        w = np.zeros((x.shape[0], self.num_topics))
        for i in range(x.shape[0]):
            for j in range(self.num_topics):
                w[i, j] = np.exp(z[i,j] - d[i])
        if(np.any(w == np.nan)):
            print "Error Banana"
            exit(1)
        for i in range(x.shape[0]):
            w[i] = w[i]/np.sum(w[i])
        if(np.any(w == np.nan)):
            print "Error Cherry"
            exit(1)

        return w

    def get_top_words(self, topx):
        top_words = np.zeros((self.num_topics, topx), dtype=object)
        for j in range(self.num_topics):
            temp_id   = self.pvec[j].argsort()[::-1][:topx]
            temp_pvec = self.pvec[j][temp_id]
            for k in range(topx):
                top_words[j,k] = (temp_id[k], temp_pvec[k])
        return top_words

    # this function returns a 30x10 matrix "aka" the top ten words(10) for each topic(30)
    def get_top_words_old(self, topx):
        top_words = np.zeros((2, 30,topx))
        for j in range(self.num_topics):
            # self.pvec[j].argsort() : returns array of indexes corresponding from smallest to largest word probabilities for topic j
            # self.pvec[j].argsort()[::-1]  : returns array of indexes corresponding from largest to smallest word probabilities for topic j
            # self.pvec[j].argsort()[::-1][:10]  : returns array of indexes corresponding from largest to smallest(max 10) word probabilities for topic j
            top_words[0, j] = self.pvec[j].argsort()[::-1][:topx]
            top_words[1, j] = self.pvec[j][top_words[0,j].astype(int)]

        return top_words

## Reading the data
docword_path = "./docword.nips.txt"
vocab_path   = "./vocab.nips.txt"

docword = open(docword_path, "r")
vocab   = open(vocab_path,   "r")

num_documents = int(docword.readline())
num_words     = int(docword.readline())
num_entries   = int(docword.readline())



dictionary = np.zeros(num_words, dtype=object)
for i in range(num_words):
    dictionary[i] = vocab.readline()



data = np.zeros((num_documents, num_words))
for i in range(num_entries):
    doc_info = map(int, docword.readline().split())
    data[doc_info[0] - 1 , doc_info[1] - 1] = doc_info[2]


total_words = data.sum(axis=0)

subset_total_words = total_words[total_words < 200]
fig = plt.figure()
plt.hist(subset_total_words, bins="auto")
plt.xlabel("word that appears $x$ times in all of the documents")
plt.ylabel("number of words in the vocabulary")
plt.title("frequency distribution of how many common/rare words there are")
plt.savefig("word_occurence_frequencies")
plt.close(fig)

excluding_total_words = total_words < 15
dictionary = dictionary[-excluding_total_words]
data = data[:, -excluding_total_words]
num_documents = data.shape[0]
num_words = data.shape[1]


print("Dictionary shape:" + str(dictionary.shape))
print("Data shape: " + str(data.shape))

"""
data = data[:SUBSET_DOC, :SUBSET_WORDS]
num_documents = data.shape[0]
num_words = data.shape[1]
"""


print("number of documents (i): " + str(num_documents))
print("number of topics (j): " + str(num_topics))
print("number of words (k): " + str(num_words))

## Initial conditions
theta = TopicTheta(num_topics, data)
for iteration in range(num_iterations):
    print("====Starting iteration " + str(iteration))
    w = theta.get_w(data)
    temp_pi = np.zeros(num_topics)
    for j in range(num_topics):
        temp_pi[j] = np.sum(w[:,j])/num_documents
    #print("pi-diff" + str(theta.pi - temp_pi))
    theta.pi = temp_pi


    temp_pvec = np.zeros((num_topics, num_words))
    for j in range(num_topics):
        temp_num = np.zeros(num_words)
        for i in range(num_documents):
            temp_num += data[i] * w[i,j]
        temp_den = 0.
        for i in range(num_documents):
            temp_den += data[i].sum() * w[i,j]
        temp_pvec[j] = temp_num/temp_den
    temp_pvec[temp_pvec == 0.] = epsilon
    #print("pvec-diff" + str(theta.pvec -temp_pvec))
    theta.pvec = temp_pvec


    fig = plt.figure()
    plt.bar(range(num_topics), theta.pi)
    plt.xlabel("Topic #")
    plt.ylabel("$\pi_j$")
    plt.title("Probability of choosing a topic - " + str(iteration))
    plt.savefig("./charts/nips-test-" + str(iteration))
    plt.close(fig)

top_words_csv = open('top_words.csv', 'w')
top_words_csv.write("topic, ")
for i in range(top_x_words):
    top_words_csv.write(str(i+1) + "th word and probability, ")
top_words_csv.write("\n")


top_words = theta.get_top_words(top_x_words)
for j in range(top_words.shape[0]):
    top_words_csv.write(str(j+1) + ", ")
    for k in range(top_words.shape[1]):
        temp_id, temp_p = top_words[j,k]
        top_words_csv.write(dictionary[temp_id].rstrip() + " " + str(round(temp_p,3)) + ", ")
    top_words_csv.write("\n")

top_words_csv.close()
