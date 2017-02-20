#! env python
import numpy as np
import matplotlib.pyplot as plt

folder_path = "./cifar-10-batches-py/"

subset = 10000

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def get_split_data(data):
    split_data = [0]*num_labels
    for label_i in range(num_labels):
        split_data[label_i] = data[labels == label_i]
    return split_data

def get_data(batch_file):
    batch = unpickle(folder_path + batch_file)
    data = np.array(batch["data"], dtype=np.float32)[0:subset]
    labels = np.array(batch["labels"])[0:subset]
    return data, labels

def show_pic(data, name):
    pic = np.zeros((32,32,3))
    for i in range(32):
        for j in range(32):
            for k in range(3):
                pic[i,j,k] = data[1024*k + 32*j + i]/256.
    plt.imsave(fname=name, arr=pic)
def get_mean(working_set):
    num_dims = len(working_set[0])
    mean = np.zeros(num_dims)
    for d in range(num_dims):
        mean[d] = np.mean(working_set[:, d])
    return mean

def get_d(means):
    N = len(means)
    d = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            mean_diff = means[i] - means[j]
            d[i,j] = np.dot(mean_diff, mean_diff)
    return d

meta_info = unpickle(folder_path + "batches.meta")
label_dict = meta_info["label_names"]
num_labels = len(label_dict)
num_dims = meta_info["num_vis"]
num_data = meta_info["num_cases_per_batch"]

data, labels = get_data("data_batch_1")
split_data = get_split_data(data)

means = np.zeros((num_labels, num_dims))
for label_i in range(num_labels):
    working_set = split_data[label_i]
    means[label_i] = get_mean(working_set)

N = num_labels

A = np.identity(N) - 1./N * np.ones((N,N))
D = get_d(means)
W = -0.5 * A.dot(D).dot(A.T)

eival, eivec = np.linalg.eig(W)

idx = eival.argsort()[::-1]
eival = eival[idx]
eivec = eivec[:,idx]

eival_r = np.diag(eival[0:2])**0.5
U_r = eivec[:,0:2]





V_T = eival_r.dot(U_r.T)
plt.plot(V_T[0], V_T[1], 'b.')
plt.show()
