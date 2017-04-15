#! env python
import numpy as np
import numpy.random as nprand
from mnist import MNIST
np.set_printoptions(threshold=np.nan)
firstx = 500
noise_pct= 0.02
width = 28
theta = 0.2

def display(data):
    for i in range(width):
        for j in range(width):
            if(data[i,j] == 1.):
                print "@",
            else:
                print ".",
        print ""
def calc_num(pi_old, noisy_images


mndata = MNIST('./data')
images, labels = mndata.load_training()
images = np.array(images[:firstx])
labels = np.array(images[:firstx])
bin_images = (images > 255./2)*2-1


noise = (nprand.rand(bin_images.shape[0], bin_images.shape[1]) > noise_pct) * 2 - 1

noisy_images = np.multiply(bin_images, noise)
noisy_images = noisy_images.reshape((-1, width, width))

pi_old = np.ones((width,width))/2.

def get_new_pi(old_pi, image):
    for x in range(width):
        for y in range(width):
            num_sum = 0.
            if(x > 0.):
                num_sum += (theta * (2.*pi_old[x-1,y]-1.) + theta * noisy_images[x,y])
            if(x < width-1):
                num_sum += (theta * (2.*pi_old[x+1,y]-1.) + theta * noisy_images[x,y])
            if(y > 0.):
                num_sum += (theta * (2.*pi_old[x,y-1]-1.) + theta * noisy_images[x,y])
            if(y < width-1):
                num_sum += (theta * (2.*pi_old[x,y+1]-1.) + theta * noisy_images[x,y])

def denoise_image(noisy_image):
    for iteration in range(10):
        new_pi = get_new_pi(old_pi, image)

for image_idx in range(noisy_images.shape[0]):
    denoise_image(noisy_images[image_idx])
