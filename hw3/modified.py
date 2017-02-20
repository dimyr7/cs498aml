#! env python
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
folder_path = "./cifar-10-batches-py/"
num_pcas = 20
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
meta_info = unpickle(folder_path + "batches.meta")
label_dict = meta_info["label_names"]
num_labels = len(label_dict)
num_dims = meta_info["num_vis"]
num_data = meta_info["num_cases_per_batch"]

def show_pic(data, name):
    pic = np.zeros((32,32,3))
    for i in range(32):
        for j in range(32):
            for k in range(3):
                pic[i,j,k] = data[1024*k + 32*j + i]/256.
    plt.imsave(fname=name, arr=pic)

def get_split_data(data):
    split_data = [0]*num_labels
    for label_i in range(num_labels):
        split_data[label_i] = data[labels == label_i]
    return split_data

def get_data(batch_file):
    batch = unpickle(folder_path + "data_batch_1")
    data = np.array(batch["data"], dtype=np.float32)[0:1000]
    labels = np.array(batch["labels"])[0:1000]
    return data, labels

def get_centered_working_set(working_set, label_index):
    mean_i = np.mean(working_set, 0)
    show_pic(mean_i, "mean" + str(label_index) + ".png")      # plot the mean
    centered = working_set - mean_i
    return centered

def get_approx_data(working_set, eigen_vec, label_index):
    centered_working_set = get_centered_working_set(working_set, label_index)
    rotated_working_set = np.zeros(centered_working_set.shape)
    for data_i in range(len(working_set)):
        rotated_working_set[data_i] = eigen_vec.T.dot(centered_working_set[data_i])

    approx_data = np.zeros(centered_working_set.shape)
    approx_data[:, 0:num_pcas] = rotated_working_set[:, 0:num_pcas]

    return rotated_working_set, approx_data

def get_sorted_eigvec(working_set):
    cov_mat = np.cov(working_set.T)
    eival, eivec = np.linalg.eig(cov_mat)

    eig_idx = eival.argsort()[::-1]
    eivec = eivec[:, eig_idx]

    return eivec


## Data preprocessing

data, labels = get_data("data_batch_1")
split_data = get_split_data(data)

error = [0]*num_labels
for label_i in range(num_labels):
    break
    print "working on " + label_dict[label_i]
    working_set = split_data[label_i]
    N = len(working_set)

    eivec = get_sorted_eigvec(working_set);

    rotated_working_set , approx_data = get_approx_data(working_set, eivec, label_i)

    error_sum = 0.
    for data_i in range(N):
        the_diff = (rotated_working_set[data_i] - approx_data[data_i])
        error_diff = (np.linalg.norm(the_diff)/np.linalg.norm(working_set[data_i]))**2
        error_sum += error_diff

    error[label_i] = (error_sum/N)



plt.figure()
error = np.zeros(10)
plt.title("Errors as a function of category")
categories = tuple(label_dict)
plt.bar(np.arange(len(categories)), error, align='center')
y_pos = np.arange(len(categories))
plt.xlabel("Label")
plt.xticks(y_pos, categories, rotation=20)
plt.ylabel("Error")
plt.savefig("parta-errors.png")
