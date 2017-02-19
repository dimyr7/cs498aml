import numpy as np
import matplotlib.pyplot as plt

folder_path = "./cifar-10-batches-py/"

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
    data = np.array(batch["data"], dtype=np.float32)[0:1000]
    labels = np.array(batch["labels"])[0:1000]
    return data, labels

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
            mean_length = len(means[i])
            mean_i = means[i].reshape((mean_length,1))
            mean_j = means[j].reshape((mean_length,1))
            d[i,j] = np.dot(means[i], means[j])
    return d

meta_info = unpickle(folder_path + "batches.meta")
label_dict = meta_info["label_names"]
num_labels = len(label_dict)
num_dims = meta_info["num_vis"]
num_data = meta_info["num_cases_per_batch"]

data, labels = get_data("data_batch_1")
split_data = get_split_data(data)

means = np.zeros((num_data, num_dims))
for label_i in range(num_labels):
    working_set = split_data[label_i]
    means[label_i] = get_mean(working_set)

N = num_data

I = np.identity(N)
ones = np.ones(N)
ones_T = np.ones(N).reshape((N, 1))
A = I - 1/float(N) * ones * ones_T
D = get_d(means)
W = 0.5 * A * D * A.T

cov_mat = np.cov(W)
eival, eivec = np.linalg.eig(cov_mat)

idx = eival.argsort()[::-1]
eival = eival[idx]
eivec = eivec[:,idx]

diag = np.array([[eival[0], 0], [0, eigval[1]]])
diag = diag**0.5

U = eivec[:2]

V_T = diag * U.T

plt.plot(V_T[0], V_T[1])
plt.show()
