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


mndata = MNIST('./data')
images, labels = mndata.load_training()
images = np.array(images[:firstx])
labels = np.array(images[:firstx])
bin_images = (images > 255./2)*2-1


noise = (nprand.rand(bin_images.shape[0], bin_images.shape[1]) > noise_pct) * 2 - 1

noisy_images = np.multiply(bin_images, noise)
noisy_images = noisy_images.reshape((-1, width, width))

pi_old = np.ones((width,width))/2.

def get_numerator(pi_old, noisy_image, (x,y)):
    num_exp_sum = 0.
    if(x > 0.):
        num_exp_sum += (theta * (2.*pi_old[x-1,y]-1.) + theta * noisy_image[x,y])
    if(x < width-1):
        num_exp_sum += (theta * (2.*pi_old[x+1,y]-1.) + theta * noisy_image[x,y])
    if(y > 0.):
        num_exp_sum += (theta * (2.*pi_old[x,y-1]-1.) + theta * noisy_image[x,y])
    if(y < width-1):
        num_exp_sum += (theta * (2.*pi_old[x,y+1]-1.) + theta * noisy_image[x,y])
    return np.exp(num_exp_sum)
def get_denomenator(pi_old, noisy_image, (x,y)):
    den_exp_sum_left = 0.
    if(x > 0.):
        den_exp_sum_left += (theta * (2.*pi_old[x-1,y]-1.) + theta * noisy_image[x,y])
    if(x < width-1):
        den_exp_sum_left += (theta * (2.*pi_old[x+1,y]-1.) + theta * noisy_image[x,y])
    if(y > 0.):
        den_exp_sum_left += (theta * (2.*pi_old[x,y-1]-1.) + theta * noisy_image[x,y])
    if(y < width-1):
        den_exp_sum_left += (theta * (2.*pi_old[x,y+1]-1.) + theta * noisy_image[x,y])

    den_exp_sum_right = 0.
    if(x > 0.):
        den_exp_sum_right += -1. * (theta * (2.*pi_old[x-1,y]-1.) + theta * noisy_image[x,y])
    if(x < width-1):
        den_exp_sum_right += -1. * (theta * (2.*pi_old[x+1,y]-1.) + theta * noisy_image[x,y])
    if(y > 0.):
        den_exp_sum_right += -1. * (theta * (2.*pi_old[x,y-1]-1.) + theta * noisy_image[x,y])
    if(y < width-1):
        den_exp_sum_right += -1. * (theta * (2.*pi_old[x,y+1]-1.) + theta * noisy_image[x,y])
    return np.exp(den_exp_sum_left) + np.exp(den_exp_sum_right)



def get_new_pi(old_pi, image):
    new_pi = np.zeros((width, width))
    for x in range(width):
        for y in range(width):
            numerator = get_numerator(pi_old, image, (x,y))
            denominator = get_denomenator(pi_old, image, (x,y))
            new_pi[x,y]= numerator/denominator
    return new_pi


def denoise_image(image):
    old_pi = np.ones((width, width))/5.
    for iteration in range(100):
        new_pi = get_new_pi(old_pi, image)
        old_pi = new_pi
    print new_pi[0]
    new_image = (new_pi > 0.5)+0.
    return new_image

new_image = denoise_image(noisy_images[0])
display(noisy_images[0])
print "======"
display(new_image)
exit(1)

for image_idx in range(noisy_images.shape[0]):
    denoise_image(noisy_images[image_idx])
