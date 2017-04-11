import numpy as np
import numpy.random as rand
import numpy.linalg as npla
import scipy.spatial.distance as dist
import scipy.misc as misc
import sklearn.mixture
import sklearn.cluster
class Image(object):
    def __init__(self, path):
        temp_img = misc.imread(path)

        self.xsize = temp_img.shape[0]
        self.ysize = temp_img.shape[1]
        self.Pixel_Size = temp_img.shape[2]
        self.data = temp_img.reshape((-1, self.Pixel_Size))/255.
        self.scalar = sklearn.preprocessing.StandardScaler()
        self.data = self.scalar.fit_transform(self.data)*10.

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
        diff = dist.cdist(self.image_data, self.mu)
        dmin = diff.min(axis=1).reshape((1,-1)).repeat(self.num_segments, axis=0).T
        ret = np.exp(-0.5 * (np.power(diff,2) - np.power(dmin,2)))
        pi = np.repeat([self.pi.T], self.image_data.shape[0], axis=0)
        ret = np.multiply(ret, pi)
        ret = np.divide(ret.T, ret.sum(axis=1)).T
        return ret



    def update_mu_pi(self, w):
        numerator = np.dot(w.T, self.image_data)
        denom= np.dot(w.T, np.ones(self.image_data.shape[0]))
        self.mu = np.divide(numerator.T, denom).T
        self.pi = denom/self.image_data.shape[0]
