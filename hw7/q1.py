#! env python
import numpy as np
import numpy.random as nprand
import scipy.misc as spmisc
from mnist import MNIST
from threading import Thread, Lock
from Queue import Queue, Empty
from time import sleep
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import boltzmann
np.set_printoptions(threshold=np.nan)
firstx = 500
noise_pct = 0.02
width = 28
c_values = np.arange(-1, 1.1, 0.4)
theta_hx = 2.

def save_image(name, data):
    spmisc.imsave(name, (data+1.)/2.)

def get_true_false_positive_rate(original, noisy, denoise):
    change  = (noisy != denoise)
    should  = (original != noisy)

    true_positive  = np.sum(np.logical_and(change, should))/float(np.sum(should))
    false_positive = np.sum(np.logical_and(change, np.logical_not(should)))/float(np.sum(np.logical_not(should)))
    return true_positive, false_positive
def start():
    while not q.empty():
        try:
            image_idx, c_idx = q.get(block=False)
            orig_image = bin_images[image_idx]
            noisy_image = noisy_images[image_idx]
            theta_hh = c_values[c_idx]
        except Empty:
            break
        machine = boltzmann.Boltzmann(width, theta_hh, theta_hx)
        denoised_image = machine.denoise_image(noisy_image)
        denoised_images[c_idx, image_idx] = denoised_image
        tp_rate, fp_rate = get_true_false_positive_rate(orig_image, noisy_image, denoised_image)


        lock.acquire()
        true_positives[c_idx, image_idx] = tp_rate
        false_positives[c_idx, image_idx] = fp_rate

        # TODO: check if this still works after doing different values of c
        pixels_correct[c_idx, image_idx] = np.sum(orig_image == denoised_image)
        if(image_idx%25 == 0):
            print "done with image:  %s, theta_hh: %s" % (str(image_idx), str(theta_hh))
        lock.release()

mndata = MNIST('./data')
images, labels = mndata.load_training()
images = np.array(images[:firstx]).reshape(-1, width,width)
labels = np.array(images[:firstx])
bin_images = (images > 255./2)*2-1


noise = (nprand.rand(bin_images.shape[0], bin_images.shape[1], bin_images.shape[2]) > noise_pct) * 2 - 1
noisy_images = np.multiply(bin_images, noise)
denoised_images = np.zeros(c_values.shape + noisy_images.shape)
lock = Lock()
q = Queue()
true_positives = np.zeros((len(c_values), firstx))
false_positives = np.zeros((len(c_values), firstx))
pixels_correct = np.zeros((len(c_values), firstx))

for c in range(len(c_values)):
    for i in range(firstx):
        q.put((i, c))

num_threads = 20

for i in range(num_threads):
    t = Thread(target=start)
    t.daemon = True
    t.run()

while not q.empty():
    sleep(1)

best_idx = pixels_correct.argmax(axis=1)
worst_idx = pixels_correct.argmin(axis=1)
for i in range(len(c_values)):
    save_image("output/images/" + str(c_values[i])+ "_best_orig.png", bin_images[best_idx[i]])
    save_image("output/images/" + str(c_values[i])+ "_best_noisy.png", noisy_images[best_idx[i]])
    save_image("output/images/" + str(c_values[i])+ "_best_denoise.png", denoised_images[i, best_idx[i]])

    save_image("output/images/" + str(c_values[i])+ "_worst_orig.png", bin_images[worst_idx[i]])
    save_image("output/images/" + str(c_values[i])+ "_worst_noisy.png", noisy_images[worst_idx[i]])
    save_image("output/images/" + str(c_values[i])+ "_worst_denoise.png", denoised_images[i, worst_idx[i]])

pixels_correct = pixels_correct/float(width*width)
pixels_cirrect_avg = np.average(pixels_correct, axis=1)
plt.figure()
plt.title("Accuracy as a function of $\\theta(H_i,H_j)$")
plt.xlabel("$\\theta(H_i,H_j)$")
plt.ylabel("Average numver of pixels correct across the entire dataset")
plt.plot(c_values, pixels_cirrect_avg, "b.")
plt.savefig("output/accuracy.png")



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

avg_tp, avg_fp= get_true_false_positive_avg(false_positives, true_positives)

plt.figure()
plt.title("Average rates as a function of $\\theta(H_i,H_k)$")
plt.xlabel("$\\theta(H_i,H_j)$")
plt.ylabel("Percent of all pixels")
plt.plot(c_values, avg_tp, "b.", label="true-positive")
plt.plot(c_values, avg_fp, "g.", label="false-positive")
plt.legend(loc=0, borderaxespad=0.)
plt.savefig("output/TP_FP_rates.png")

plt.figure()
plt.title("ROC")
plt.xlabel("False Positive")
plt.ylabel("True Positive");
plt.plot(avg_fp, avg_tp,  "r.")
plt.savefig("output/ROC.png")

# plot_rates()
