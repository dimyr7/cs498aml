#! env python
import numpy as np
import numpy.random as nprand
import scipy.misc as spmisc
from mnist import MNIST
from threading import Thread, Lock
from Queue import Queue, Empty
from time import sleep
from matplotlib.pyplot import plot, show
np.set_printoptions(threshold=np.nan)
firstx = 500
noise_pct = 0.02
width = 28
c_values = np.arange(-1, 1.1, 0.1)
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

def get_numerator(pi_old, noisy_image, (x,y), theta_hh):
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
def get_denomenator(pi_old, noisy_image, (x,y), theta_hh):
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



def get_pi_new(pi_old, image, c):
    pi_new = np.zeros((width, width))
    for x in range(width):
        for y in range(width):
            numerator = get_numerator(pi_old, image, (x,y), c)
            denominator = get_denomenator(pi_old, image, (x,y), c)
            pi_new[x,y]= numerator/denominator
    return pi_new


def denoise_image(image, c):
    pi_old = np.ones((width, width))/5.
    for iteration in range(10):
        pi_new = get_pi_new(pi_old, image, c)
        pi_old = pi_new
    new_image = (pi_new > 0.5) * 2 - 1
    return new_image

lock = Lock()
q = Queue()
true_positives = np.zeros((len(c_values), bin_images.shape[0]))
false_positives = np.zeros((len(c_values), bin_images.shape[0]))

best_idx = 0.
best_rate = np.PINF
best_denoised = np.zeros((width, width))
worst_idx = 0.
worst_rate = np.NINF
worst_denoised = np.zeros((width, width))

def start():
    global best_rate
    global best_denoised
    global best_idx

    global worst_rate
    global worst_denoised
    global worst_idx

    while not q.empty():
        try:
            v = q.get(block=False)
            image_idx = v[0]
            c_idx = v[1]
            c = c_values[c_idx]
        except Empty:
            break
        denoised_image = denoise_image(noisy_images[image_idx], c)
        t, f = get_true_false_positive_rate(bin_images[image_idx], noisy_images[image_idx], denoised_image)
        lock.acquire()
        true_positives[c_idx, image_idx] = t
        false_positives[c_idx, image_idx] = f

        # TODO: check if this still works after doing different values of c
        curr_error_rate = np.sum(bin_images[image_idx] != denoised_image)/(width*width)
        if(curr_error_rate < best_rate):
            best_rate = curr_error_rate
            best_idx = image_idx
            best_denoised = denoised_image
        elif(curr_error_rate > worst_rate):
            worst_rate = curr_error_rate
            worst_idx = image_idx
            worst_denoised = denoised_image
        print "done with image:  %s, c: %s" % (str(image_idx), str(c))
        lock.release()

for c in range(len(c_values)):
    for i in range(bin_images.shape[0]):
        q.put((i, c))

num_threads = 20

for i in range(num_threads):
    t = Thread(target=start)
    t.daemon = True
    t.run()

while not q.empty():
    sleep(1)

def plot_rates():
    global false_positives
    global true_positives
    ind = false_positives.argsort()
    false_positives = false_positives[ind[::1]]
    true_positives = true_positives[ind[::1]]
    print "false positives: " + str(false_positives)
    print "true positives: " + str(true_positives)
    p = plot(false_positives, true_positives)
    show(p)

def get_true_false_positive_avg(false_positives, true_positives):
    avg_false_positives = np.zeros(false_positives.shape[0])
    avg_true_positives = np.zeros(false_positives.shape[0])

    for i in range(false_positives.shape[0]):
        avg_true_positives[i] = np.average(true_positives[i])
        avg_false_positives[i] = np.average(false_positives[i])

    return avg_true_positives, avg_false_positives

avg_true, avg_false = get_true_false_positive_avg(false_positives, true_positives)

print avg_true
print avg_false

p = plot(avg_false, avg_true, marker="o")
show(p)

# plot_rates()

save_image("best_original.png", bin_images[best_idx])
save_image("best_noisy.png", noisy_images[best_idx])
save_image("best_denoised.png", best_denoised)

save_image("worst_original.png", bin_images[worst_idx])
save_image("worst_noisy.png", noisy_images[worst_idx])
save_image("worst_denoised.png", worst_denoised)
