#! env python
import numpy as np
import numpy.random as nprand
import scipy.misc as spmisc
from mnist import MNIST
np.set_printoptions(threshold=np.nan)
firstx = 500
noise_pct= 0.5
width = 28
theta_hh = 0.2
theta_hx = 2.

def display(data):
    for i in range(width):
        for j in range(width):
            if(data[i,j] == 1.):
                print "@",
            else:
                print " ",
        print ""
def save_image(name, data):
    spmisc.imsave(name, (data+1.)/2.)


def get_true_false_positive_rate(original, noisy, denoise):
    # o -> original, n -> new, d -> denoise
    def change(i, j):
        return noisy[i,j] != denoise[i,j]

    def should(i, j):
        return original[i,j] != noisy[i,j]

    true_positive = 0
    false_positive = 0

    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            if change(i,j) == should(i,j):
                true_positive += 1
            elif change is True and should is False:
                false_positive += 1

    total_change = true_positive + false_positive

    if total_change == 0:
        return 0,0

    return true_positive / float(total_change), false_positive / float(total_change)



mndata = MNIST('./data')
images, labels = mndata.load_training()
images = np.array(images[:firstx]).reshape(-1, width,width)
labels = np.array(images[:firstx])
bin_images = (images > 255./2)*2-1


noise = (nprand.rand(bin_images.shape[0], bin_images.shape[1], bin_images.shape[2]) > noise_pct) * 2 - 1

noisy_images = np.multiply(bin_images, noise)


def get_numerator(pi_old, noisy_image, (x,y)):
    num_exp_sum = 0.
    if(x > 0):
        num_exp_sum += (theta_hh * (2.*pi_old[x-1,y]-1.) + theta_hx * noisy_image[x-1,y])
    if(x < width-1):
        num_exp_sum += (theta_hh * (2.*pi_old[x+1,y]-1.) + theta_hx * noisy_image[x+1,y])
    if(y > 0):
        num_exp_sum += (theta_hh * (2.*pi_old[x,y-1]-1.) + theta_hx * noisy_image[x,y-1])
    if(y < width-1):
        num_exp_sum += (theta_hh * (2.*pi_old[x,y+1]-1.) + theta_hx * noisy_image[x,y+1])
    return np.exp(num_exp_sum)
def get_denomenator(pi_old, noisy_image, (x,y)):
    den_exp_sum_left = 0.
    if(x > 0):
        den_exp_sum_left += (theta_hh * (2.*pi_old[x-1,y]-1.) + theta_hx * noisy_image[x-1,y])
    if(x < width-1):
        den_exp_sum_left += (theta_hh * (2.*pi_old[x+1,y]-1.) + theta_hx * noisy_image[x+1,y])
    if(y > 0):
        den_exp_sum_left += (theta_hh * (2.*pi_old[x,y-1]-1.) + theta_hx * noisy_image[x,y-1])
    if(y < width-1):
        den_exp_sum_left += (theta_hh * (2.*pi_old[x,y+1]-1.) + theta_hx * noisy_image[x,y+1])

    den_exp_sum_right = 0.
    if(x > 0):
        den_exp_sum_right += -1. * (theta_hh * (2.*pi_old[x-1,y]-1.) + theta_hx * noisy_image[x-1,y])
    if(x < width-1):
        den_exp_sum_right += -1. * (theta_hh * (2.*pi_old[x+1,y]-1.) + theta_hx * noisy_image[x+1,y])
    if(y > 0):
        den_exp_sum_right += -1. * (theta_hh * (2.*pi_old[x,y-1]-1.) + theta_hx * noisy_image[x,y-1])
    if(y < width-1):
        den_exp_sum_right += -1. * (theta_hh * (2.*pi_old[x,y+1]-1.) + theta_hx * noisy_image[x,y+1])
    return np.exp(den_exp_sum_left) + np.exp(den_exp_sum_right)



def get_pi_new(pi_old, image):
    pi_new = np.zeros((width, width))
    for x in range(width):
        for y in range(width):
            numerator = get_numerator(pi_old, image, (x,y))
            denominator = get_denomenator(pi_old, image, (x,y))
            pi_new[x,y]= numerator/denominator
    return pi_new


def denoise_image(image):
    pi_old = np.ones((width, width))/5.
    for iteration in range(10):
        pi_new = get_pi_new(pi_old, image)
        pi_old = pi_new
    new_image = (pi_new > 0.5) * 2 - 1
    return new_image

denoise_images = np.zeros(noisy_images.shape)
for image_idx in range(noisy_images.shape[0]):
    denoise_images[image_idx] = denoise_image(noisy_images[image_idx])

rates = np.array((noisy_images.shape[0]))
for image_idx in range(noisy_images.shape[0]):
    print get_true_false_positive_rate(bin_images[image_idx], noisy_images[image_idx], denoise_images[image_idx])

