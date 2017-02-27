#! env python
import numpy as np
import matplotlib.pyplot as plt

folder_path = "./cifar-10-batches-py/"

subset = 10000
num_pcas = 20

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


def get_sorted_eigvec(working_set):
    cov_mat = np.cov(working_set.T)
    eival, eivec = np.linalg.eig(cov_mat)

    eig_idx = eival.argsort()[::-1]
    eivec = eivec[:, eig_idx]

    return eivec

meta_info = unpickle(folder_path + "batches.meta")
label_dict = meta_info["label_names"]
num_labels = len(label_dict)
num_dims = meta_info["num_vis"]
num_data = meta_info["num_cases_per_batch"]

num_batches = 2
Vs = [0] * num_batches
for batch in range(num_batches):
    """
    Working for each batch
    """
    print "== working on batch " + str(batch) + " =="
    data, labels = get_data("data_batch_1")
    split_data = get_split_data(data)

    """
    For each category, get all means and PCAs
    """
    means = np.zeros((num_labels, num_dims))
    PCAs = [0]*num_labels
    print "===== Calculating means ====="
    for label_i in range(num_labels):
        print "working on " + label_dict[label_i]
        working_set = split_data[label_i]
        means[label_i] = np.mean(working_set, 0)
        centered_working_set = working_set - means[label_i]
        split_data[label_i] = centered_working_set


        PCAs[label_i] = get_sorted_eigvec(centered_working_set)

    print "===== Calculating E ====="

    E = np.zeros((num_labels, num_labels))
    """
    Entry E[i , j] represents the quantity E(i -> j)
    """
    for label_i in range(num_labels):
        for label_j in range(num_labels):
            working_set = np.copy(split_data[label_i])
            N = len(working_set)
            print "working on ( " + label_dict[label_i] + " -> " + label_dict[label_j] + " )"
            """
            Using data with label i but the PCAs from label j
            """
            eivec = PCAs[label_j]
            rotated_working_set = np.zeros(working_set.shape)
            for data_i in range(N):
                rotated_working_set[data_i] = np.dot(eivec.T, working_set[data_i])
            approx_data = np.zeros(rotated_working_set.shape)
            approx_data[: , 0:num_pcas] = rotated_working_set[:, 0:num_pcas]

            error_sum = 0.
            for data_i in range(N):
                the_diff = (approx_data[data_i] - rotated_working_set[data_i])
                error_diff = (np.linalg.norm(the_diff)/np.linalg.norm(rotated_working_set[data_i]))**2
                error_sum += error_diff
            if(label_i == label_j):
                print (error_sum/N)
            E[label_i, label_j] = (error_sum/N)

    print E
    """
    Forming matrices A, D, W as in the book
    """
    A = np.identity(num_labels) - 1./num_labels * np.ones((num_labels, num_labels))
    D = np.zeros((num_labels, num_labels))
    for label_i in range(num_labels):
        for label_j in range(num_labels):
            """
            E(A -> A) will not be 0 because using only 20 prinicipal compoents will result in atleast a small amount of error
            """
            D[label_i, label_j] = 0.5 * (E[label_i, label_j] + E[label_j, label_i])
    W = -0.5 * A.dot(D).dot(A.T)

    """
    Continuing process from the book
    """
    eival, eivec = np.linalg.eig(W)
    idx = eival.argsort()[::-1]
    eivec = eivec[:,idx]

    eival_r = np.diag(eival[0:2])**0.5
    Ur_T = eivec[:,0:2].T

    V_T = eival_r.dot(Ur_T)
    Vs[batch] = V_T

"""
Average the coordinates from 5 different batches
"""
V_T = np.mean(Vs, 0)
fig, ax = plt.subplots()
ax.scatter(V_T[0], V_T[1])
plt.title("2D Map of Categories using modified PCoA")

for i, txt in enumerate(label_dict):
    ax.annotate(txt, (V_T[0,i], V_T[1,i]))
fig.savefig("part-c map.png")
