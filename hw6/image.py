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
    def __init__(self, path=None, orig=None):
        if(orig != None):
            self.copy_constructor(orig)
        elif(path != None):
            temp_img = misc.imread(path)

            self.xsize = temp_img.shape[0]
            self.ysize = temp_img.shape[1]
            self.Pixel_Size = temp_img.shape[2]
            self.data = temp_img.reshape((-1, self.Pixel_Size))/RGB_Range

    def save_pic(self, name):
        print "saving pic to " + str(name)
        pic = self.data.reshape((self.xsize, self.ysize, self.Pixel_Size))
        plt.imsave(fname=name, arr=pic)
    def copy_constructor(self, orig):
        self.xsize = orig.xsize
        self.ysize = orig.ysize
        self.Pixel_Size = orig.Pixel_Size
        self.data = np.copy(orig.data)



class NormalTheta(object):
    def __init__(self, num_segments, x):
        self.num_segments = num_segments
        self.mu = np.zeros((num_segments, x.shape[1]))
        self.pi = np.zeros(num_segments)

        #pi[j] --> probability that the cluster is chosen
        #mu[j] --> mean center value of given cluster j
        # given a pixel, what cluster is chosen ?

        kmeans = sklearn.cluster.KMeans(n_clusters = num_segments).fit(x)
        #determine pi[j]
        for j in range(num_segments):
            self.pi[j] = (kmeans.labels_ == j).sum()/float(x.shape[0])

        #determine mu[j]
        self.mu = kmeans.cluster_centers_
        """
        self.mu = rand.rand(num_segments, x.shape[1])
        self.pi = rand.rand(num_segments)
        self.pi = self.pi/float(self.pi.sum())
        """



    def get_d(self, x):
        print "get_d"
        d_min = np.zeros(x.shape[0])
        return d_min
        d_min[:] = np.NINF
        for i in range(x.shape[0]):
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
            w[i] = w[i]/(w[i].sum())
        return w

images = {'flower'   : Image(path="./em_images/flower.png"),
          'fish' : Image(path="./em_images/fish.png"),
          'sunset' : Image(path="./em_images/sunset.png")}
def show_segments(image,theta, num_segments, path):
    log_likelihood = np.zeros((image.data.shape[0], num_segments))
    for i in range(image.data.shape[0]):
        for j in range(num_segments):
            temp_diff = image.data[i] - theta.mu[j]
            log_likelihood[i,j] = -0.5 * temp_diff.dot(temp_diff) + np.log(theta.pi[j])
    assigned_segments = log_likelihood.argmax(axis=1)


    new_image = Image(orig=image)
    for i in range(image.data.shape[0]):
        new_image.data[i] = theta.mu[assigned_segments[i]]
    new_image.save_pic(path)



def do_em((name, image), num_segments):
    theta = NormalTheta(num_segments, image.data)
    show_segments(image, theta, num_segments, "output/" + name + "_" + "kmeans.png")
    for iteration in range(num_iterations):
        print("== Starting Iteration: " + str(iteration))
        w = theta.get_w(image.data)

        print "updating mu"
        temp_mu = np.zeros((num_segments, image.data.shape[1]))
        # Update mu
        for j in range(num_segments):
            temp_sum = np.zeros(image.data.shape[1])
            for i in range(image.data.shape[0]):
                temp_sum += image.data[i] * w[i,j]
            temp_mu = temp_sum/(w[:, j]).sum()
            print temp_mu
            theta.mu[j] = temp_mu

        print "updating pi"
        temp_pi = np.zeros(num_segments)
        for j in range(num_segments):
            temp_pi[j] = w[:, j].sum()/image.data.shape[0]
        theta.pi = temp_pi
        path = "output/" + name + "_" + str(num_segments) + "_" +  str(iteration) + "_" + str(num_iterations) + ".png"
        show_segments(image, theta, num_segments, path)
        sys.stdout.flush()

"""
good_em = sklearn.mixture.GaussianMixture(n_components=10).fit(images['fish'].data)
log_likelihood = np.zeros((images['fish'].data.shape[0], 10))
for i in range(images['fish'].data.shape[0]):
    for j in range(10):
        temp_diff = images['fish'].data[i] - good_em.means_[j]
        log_likelihood[i,j] = -0.5 * temp_diff.dot(temp_diff) + np.log(good_em.weights_[j])
assigned_segments = log_likelihood.argmax(axis=1)


new_image = Image(orig=images['fish'])
for i in range(images['fish'].data.shape[0]):
    new_image.data[i] = good_em.means_[assigned_segments[i]]
new_image.save_pic("./output/fish_goodEM.png")
"""



for key, image in images.iteritems():
    print("========= Image: " + key)
    for num_segments in segments:
        print("===== Num Segs: " + str(num_segments))
        do_em((key, image), num_segments)

do_em(("partb", images["sunrise"]), 20)
