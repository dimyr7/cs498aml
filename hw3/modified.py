#! env python
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#np.set_printoptions(threshold=np.nan)
folder_path = "./cifar-10-batches-py/"
num_pcas = 20
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

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

def get_mean(working_set):
    num_dims = len(working_set[0])
    mean = np.zeros(num_dims)
    for d in range(num_dims):
        mean[d] = np.mean(working_set[:, d])
    return mean

def get_data(batch_file):
    batch = unpickle(folder_path + "data_batch_1")
    data = np.array(batch["data"], dtype=np.float32)[0:1000]
    labels = np.array(batch["labels"])[0:1000]
    return data, labels

def center_working_set(working_set, working_set_mean):
    centered_working_set = np.array(working_set)
    for i in range(len(working_set_mean)):
        centered_working_set[:, i] -= working_set_mean[i]
    return centered_working_set

def get_centered_working_set(working_set, label_index):
    mean = get_mean(working_set)                   # get working set mean
    show_pic(mean, "mean" + str(label_index))      # plot the mean
    centered = center_working_set(working_set, mean)          # center the working set
    return mean, centered

def get_approx_data(working_set, eigen_vec, label_index):
    mean, centererd_working_set = get_centered_working_set(working_set, label_index)
    sum = 0
    for i in range(num_pcas):
        x = np.dot(eigen_vec[i].T, centered_working_set)
        x = np.dot(x, eigen_vec[i])
        sum += x
    return mean + sum

meta_info = unpickle(folder_path + "batches.meta")
label_dict = meta_info["label_names"]
num_labels = len(label_dict)
num_dims = meta_info["num_vis"]
num_data = meta_info["num_cases_per_batch"]

## Data preprocessing

data, labels = get_data("data_batch_1")
split_data = get_split_data(data)

error = [0]*num_labels
for label_i in range(num_labels):
    print "working on " + label_dict[label_i]
    working_set = split_data[label_i]
    N = len(working_set)

    cov_mat = np.cov(working_set.T)
    eival, eivec = np.linalg.eig(cov_mat)

    approx_data = get_approx_data(working_set, eivec, label_i)

    error_sum = 0.
    for data_i in range(N):
        the_diff = (approx_data[data_i] - working_set[data_i])
        error_sum += (np.linalg.norm(the_diff)/np.linalg.norm(working_set[data_i]))**2
    error[label_i] = (error_sum/N)

    # Todo make this plot nice looking
    plt.figure()
    plt.title("Principal components")
    plt.xlabel("$n^{th}$ greatest eigenvalue")
    plt.ylabel("Eigenvalue")
    plt.plot(eival)
    plt.savefig("PCA" + str(label_i))

plt.figure()
plt.title("Errors as a function of category")
categories = tuple(label_dict)
plt.bar(np.arange(len(categories)), error, align='center')
y_pos = np.arange(len(categories))
plt.xlabel("Label")
plt.xticks(y_pos, categories, rotation='vertical')
plt.ylabel("Error")
plt.savefig("errors")
