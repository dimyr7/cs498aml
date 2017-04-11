#! env python
import numpy as np
import em
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

num_iterations = 10
images = np.array(["fish", "flower", "sunset"], dtype=object)
segments = np.array([10, 20, 50])


images = {'flower'   : em.Image(path="./em_images/flower.png"),
          'fish' : em.Image(path="./em_images/fish.png"),
          'sunset' : em.Image(path="./em_images/sunset.png")}

def do_em((name, image), num_segments):
    theta = em.NormalTheta(num_segments, image.data)
    w = np.zeros((image.data.shape[0], num_segments))
    for iteration in range(num_iterations):
        print("== Starting Iteration: " + str(iteration))
        w = theta.get_w()
        theta.update_mu_pi(w)

    new_means = image.scalar.inverse_transform(theta.mu/10.)
    assigned_segments = w.argmax(axis=1)
    new_data = np.zeros((image.data.shape[0], image.data.shape[1]))
    for i in range(image.data.shape[0]):
        new_data[i] = new_means[assigned_segments[i]]

    pic = new_data.reshape((image.xsize, image.ysize, image.Pixel_Size))
    path = "q2_output/" + name + "_" + str(num_segments) + ".png"
    plt.imsave(fname=path, arr=pic)
for key, image in images.iteritems():
    print("========= Image: " + key)
    for num_segments in segments:
        print("===== Num Segs: " + str(num_segments))
        do_em((key, image), num_segments)
for i in range(5):
    do_em(("partb" + str(i), images["sunset"]), 20)
