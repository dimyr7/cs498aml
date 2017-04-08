#! env python
import numpy as np
import scipy.misc as misc
import sklearn.cluster
import matplotlib as mpl
import glob
import numpy.random as rand
import sys
mpl.use('Agg')
import matplotlib.pyplot as plt
import sklearn.mixture
np.set_printoptions(threshold=np.nan)

RGB_Range = 255.
num_iterations = 10
images = np.array(["fish", "flower", "sunset"], dtype=object)
segments = np.array([10, 20, 50])

class Image(object):
    def __init__(self, path):
        temp_img = misc.imread(path)

        self.xsize = temp_img.shape[0]
        self.ysize = temp_img.shape[1]
        self.Pixel_Size = temp_img.shape[2]
        self.data = temp_img.reshape((-1, self.Pixel_Size))/RGB_Range
        self.scalar = sklearn.preprocessing.StandardScaler()
        self.data = self.scalar.fit_transform(self.data)

class NormalTheta(object):
    def __init__(self, num_segments, image_data):
        self.num_segments = num_segments
        """
        pi[j] --> probability that the cluster is chosen
        mu[j] --> mean center value of given cluster j
        """
        self.mu = rand.rand(num_segments, image_data.shape[1])
        self.pi = np.ones(num_segments)/num_segments
        self.image_data = image_data

    def get_w(self):
        print "get_w"
        w = np.zeros((self.image_data.shape[0], self.num_segments))
        for i in range(self.image_data.shape[0]):
            for j in range(self.num_segments):
                temp_diff = self.image_data[i] - self.mu[j]
                temp_exponent = -0.5 * temp_diff.dot(temp_diff)
                w[i,j] = self.pi[j] * np.exp(temp_exponent)
        for i in range(self.image_data.shape[0]):
            w[i] /= w[i].sum()
        return w

    def update_mu_pi(self, w):
        for j in range(self.num_segments):
            numerator = sum([self.image_data[i] * w[i,j] for i in range(self.image_data.shape[0])])
            wij_sum = sum([w[i,j] for i in range(self.image_data.shape[0])])
            self.mu[j] = numerator / wij_sum
            self.pi[j] = wij_sum / self.image_data.shape[0]

images = {'flower'   : Image(path="./em_images/flower.png"),
          'fish' : Image(path="./em_images/fish.png"),
          'sunset' : Image(path="./em_images/sunset.png")}

def do_em((name, image), num_segments):
    theta = NormalTheta(num_segments, image.data)
    w = np.zeros((image.data.shape[0], num_segments))
    for iteration in range(num_iterations):
        print("== Starting Iteration: " + str(iteration))
        w = theta.get_w()
        print "updating mu and pi"
        theta.update_mu_pi(w)

    new_means = image.scalar.inverse_transform(theta.mu)
    assigned_segments = w.argmax(axis=1)
    new_data = np.zeros((image.data.shape[0], image.data.shape[1]))
    for i in range(image.data.shape[0]):
        new_data[i] = new_means[assigned_segments[i]]

    pic = new_data.reshape((image.xsize, image.ysize, image.Pixel_Size))
    path = "q2_new_output/" + name + "_" + str(num_segments) + ".png"
    plt.imsave(fname=path, arr=pic)

'''
for key, image in images.iteritems():
    print("========= Image: " + key)
    for num_segments in segments:
        print("===== Num Segs: " + str(num_segments))
        do_em((key, image), num_segments)
'''

do_em(("partb", images["fish"]), 20)
