#! env python

import numpy as np
from scipy import misc
import sklearn.cluster
import matplotlib as mpl
import glob
mpl.use('Agg')
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

RGB_Range = 255
Pixel_Size = 3
num_iterations = 10
images = np.array(["fish", "flower", "sunset"], dtype=object)
segments = np.array([10, 20, 50])

class NormalTheta(object):
    def __init__(self, num_segments, x):
        self.num_segments = num_segments
        self.mu = np.zeros((num_segments, x.shape[1]))
        self.pi = np.zeros((num_segments, x.shape[1]))

        #pi[j] --> probability that the cluster is chosen
        #mu[j] --> mean center value of given cluster j
        # given a pixel, what cluster is chosen ?

        kmeans = sklearn.cluster.KMeans(n_clusters = num_segments).fit(x)

        #determine pi[j]
        for j in range(num_segments):
            self.pi[j] = (kmeans.labels_ == j).sum()/float(x.shape[0])

        #determine mu[j]
        print kmeans.cluster_centers_.shape
        for j in range(num_segments):
            self.mu[j] = kmeans.cluster_centers_[j]

        print self.pi
        print self.mu



    def get_d(self, x):
        print "get_d"
        d_min = np.zeros(x[0].shape)
        d_min[:] = np.NINF
        for i in range(x[0].shape):
            for j in range(self.num_segments):
                temp_diff = x[i] - self.mu[j]
                if(temp_diff.dot(temp_diff) > d_min[i]):
                    d_min[i] = temp_diff.dot(temp_diff)
        return d_min



    def get_w(self, x):
        print "get_w"
        d = self.get_d(x)
        w = np.zeros((x.shape[0], self.num_segments))
        for i in range(x.shape[0]):
            for j in range(self.num_segments):
                temp_diff = x[i] - self.mu[j]
                temp_exponent = -0.5 * (temp_diff.dot(temp_diff)  - d[i])
                w[i,j] = self.pi[j] * np.exp(temp_exponent)
        for i in range(x.shape[0]):
            w[i,] = w[i,]/w[i,].sum()
        return w

images = {'fish'   : misc.imread("./em_images/fish.png").reshape((-1,   Pixel_Size))/RGB_Range,
          'flower' : misc.imread("./em_images/flower.png").reshape((-1, Pixel_Size))/RGB_Range,
          'sunset' : misc.imread("./em_images/sunset.png").reshape((-1, Pixel_Size))/RGB_Range}


def do_em(data, num_segments):
    theta = NormalTheta(num_segments, data)
    print data
    exit(0)
    for iteration in range(num_iterations):
        print("== Starting Iteration: " + str(iteration))
        w = theta.get_w(data)
        temp_mu = np.zeros((num_segments, data.shape[1]))
        for j in range(num_segments):
            temp_sum = np.zeros(data.shape[1])
            for i in range(x.shape[0]):
                temp_sum += x[i] * w[i,j]
            temp_mu = temp_sum/w[:, j].sum()


        temp_pi = np.zeros(num_segments)
        for j in range(num_segments):
            temp_pi[j] = w[:, j].sum()/data.shape[0]
        theta.pi = temp_pi
    """
    TODO - save and color the resulting clusters
    """



for k, image in images.iteritems():
    print("========= Image: " + k)
    for num_segments in segments:
        print("===== Num Segs: " + str(num_segments))
        do_em(image, num_segments)

do_em(images["fish"], 20)
