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
        """
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
    w = theta.get_w(image.data)
    assigned_segments = w.argmax(axis=1)


    new_image = Image(orig=image)
    for i in range(image.data.shape[0]):
        new_image.data[i] = theta.mu[assigned_segments[i]]
    new_image.save_pic(path)



def do_em((name, image), num_segments):
    stop_criteria = 0.00001
    num_pixels = image.data.shape[0]
    height = image.xsize
    width = image.ysize

    pis = np.ones(num_segments)/num_segments
    means = rand.rand(num_segments, image.data.shape[1])
    last_Q = np.NINF
    """EM Steps"""
    while(True):
        print "E step"
        inner = np.zeros((image.data.shape[0], num_segments))
        for i in range(num_segments):
            dist = image.data-means[i]
            inner[:,i] =  (-0.5) * np.sum(np.power(dist, 2),axis=1)

        print "Calc wij"
        top = np.dot(np.exp(inner)  , np.diagflat(pis))
        bottom = top.sum(axis=1)
        wijs = ((top.T)/bottom).T


        print "Calc Q"
        Q = np.sum(np.multiply(inner, wijs))

        print "M step"
        for j in range(num_segments):
            top = np.sum((image.data.T  * wijs[:,j]).T, axis=0)
            bottom = sum(wijs[:,j])
            means[j] = top/bottom
            pis[j] = sum(wijs[:,j])/image.data.shape[0]
        diff_Q = np.abs(Q - last_Q)
        print means
        if(diff_Q < stop_criteria):
            break
        else:
            last_Q = Q
    assigned_segments = wijs.argmax(axis=1)
    new_image = Image(orig=image)
    for i in range(image.data.shape[0]):
        new_image.data[i] = means[assigned_segments[i]]
    new_image.save_pic("./output/R_" + name + "_" + str(num_segments) + ".png")

for key, image in images.iteritems():
    print("========= Image: " + key)
    for num_segments in segments:
        print("===== Num Segs: " + str(num_segments))
        do_em((key, image), num_segments)

do_em(("partb", images["sunrise"]), 20)
