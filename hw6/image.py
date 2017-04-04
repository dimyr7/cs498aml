#! env python

import numpy as np
from scipy import misc
import sklearn.cluster
import matplotlib as mpl
import glob
mpl.use('Agg')
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

num_iterations = 10
images = np.array(["fish", "flower", "sunset"], dtype=object)
segments = np.array([10, 20, 50])

class NormalTheta(object):
    def __init__(self, num_segments, x):
        self.num_segments = num_segments
        self.mu = np.zeros(num_segments)
        self.pi = np.zeros(num_segments)

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

images = {'fish'   : misc.imread("./em_images/fish.png"),
          'flower' : misc.imread("./em_images/flower.png"),
          'sunset' : misc.imread("./em_images/sunset.png")}

def do_em(data, num_segments):
    theta = NormalTheta(num_segments, data)
    for iteration in range(num_iterations):
        """
        TOOD
        """

for _, image in images.iteritems():
    for num_segments in segments:
        do_em(image, num_segments)

do_em(images["fish"], 20)
