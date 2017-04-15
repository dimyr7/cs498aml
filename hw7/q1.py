#! env python
import numpy as np
import numpy.random as nprand
import scipy.misc as spmisc
from mnist import MNIST
from threading import Thread, Lock
from Queue import Queue
from time import sleep
np.set_printoptions(threshold=np.nan)
firstx = 500
noise_pct= 0.05
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
    change  = (noisy != denoise)
    should  = (original != noisy)

    true_positive  = np.sum(np.logical_and(change, should))/float(np.sum(should))
    false_positive = np.sum(np.logical_and(change, np.logical_not(should)))/float(np.sum(np.logical_not(should)))
    return true_positive, false_positive



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

lock = Lock()
q = Queue()
rates = []

def start():
    while not q.empty():
        try:
            image_idx = q.get(block=False)
        except Queue.Empty:
            break
        denoised_image = denoise_image(noisy_images[image_idx])
        rate = get_true_false_positive_rate(bin_images[image_idx], noisy_images[image_idx], denoised_image)
        lock.acquire()
        rates.append(rate)
        print "done with image" + str(image_idx)
        lock.release()

for i in range(bin_images.shape[0]):
    q.put(i)

num_threads = 10

for i in range(num_threads):
    t = Thread(target=start)
    t.daemon = True
    t.run()

while not q.empty():
    sleep(1)

print rates

'''
orig = bin_images[0]
noisy = noisy_images[0]
new = denoise_image(noisy)

display(orig)
print "============="
display(noisy)
print "============="
display(new)
get_true_false_positive_rate(orig, noisy, new)
exit(1)



denoise_images = np.zeros(noisy_images.shape)
for image_idx in range(10):
    denoise_images[image_idx] =


rates = np.array((noisy_images.shape[0]))
for image_idx in range(10):
    print get_true_false_positive_rate(bin_images[image_idx], noisy_images[image_idx], denoise_images[image_idx])
'''

