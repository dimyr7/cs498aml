import numpy as np
import numpy.random as rand
import numpy.linalg as npla
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

    def get_xnorm(self):
        x_norm = npla.norm(self.image_data, axis=1)
        x_norm = np.power(x_norm,2)
        x_norm = x_norm.reshape((1,-1))
        x_norm = x_norm.repeat( self.num_segments, axis=0).T
        return x_norm
    def get_munorm(self):
        mu_norm = npla.norm(self.mu , axis=1)
        mu_norm = np.power(mu_norm, 2)
        mu_norm = mu_norm.reshape((1,-1))
        mu_norm = mu_norm.repeat(self.image_data.shape[0], axis=0)
        return mu_norm
    def get_w(self):
        x_norm = self.get_xnorm()
        y_norm = self.get_munorm()
        x_mu_dot = -2 * np.dot(self.image_data, self.mu.T)
        ret = np.exp(-0.5 * (x_norm + y_norm + x_mu_dot))
        pi = np.repeat([self.pi.T], self.image_data.shape[0], axis=0)
        ret = np.multiply(ret, pi)
        ret = np.divide(ret.T, ret.sum(axis=1)).T
        return ret

    def get_w_old(self):
        w = np.zeros((self.image_data.shape[0], self.num_segments))
        for i in range(self.image_data.shape[0]):
            for j in range(self.num_segments):
                diff = self.image_data[i] - self.mu[j]
                temp_exponent = -0.5 * diff.dot(diff)
                w[i,j] = self.pi[j] * np.exp(temp_exponent)
        for i in range(self.image_data.shape[0]):
            w[i] /= w[i].sum()
        return w


    def update_mu_pi(self, w):
        numerator = np.dot(w.T, self.image_data)
        denom= np.dot(w.T, np.ones(self.image_data.shape[0]))
        self.mu = np.divide(numerator.T, denom).T
        self.pi = denom/self.image_data.shape[0]
