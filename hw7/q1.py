#! env python
import numpy as np
import numpy.random as nprand
from mnist import MNIST
np.set_printoptions(threshold=np.nan)
firstx = 500
noise_pct= 0.02

def display(data):
    for i in range(28):
        for j in range(28):
            index = 28*i + j
            if(data[index] == 1.):
                print "@",
            else:
                print ".",
        print ""


mndata = MNIST('./data')
images, labels = mndata.load_training()
images = np.array(images[:firstx])
labels = np.array(images[:firstx])
bin_images = (images > 255./2)*2-1


noise = (nprand.rand(bin_images.shape[0], bin_images.shape[1]) > noise_pct) * 2 - 1

noisy_images = np.multiply(bin_images, noise)

display(bin_images[0])
print "=========="
display(noisy_images[0])

