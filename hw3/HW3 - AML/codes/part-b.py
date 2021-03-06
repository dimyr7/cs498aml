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

meta_info = unpickle(folder_path + "batches.meta")
label_dict = meta_info["label_names"]
num_labels = len(label_dict)
num_dims = meta_info["num_vis"]
num_data = meta_info["num_cases_per_batch"]

def get_split_data(data):
    """
    Splitting the the data based on the category
    """
    split_data = [0]*num_labels
    for label_i in range(num_labels):
        split_data[label_i] = data[labels == label_i]
    return split_data

def get_data(batch_file):
    batch = unpickle(folder_path + batch_file)
    data = np.array(batch["data"], dtype=np.float32)[0:subset]
    labels = np.array(batch["labels"])[0:subset]
    return data, labels

num_batches = 5
mean_batch = [0] * num_batches
for batch in range(num_batches):
    """
    Getting the mean for each category in each batch
    """
    print "working on batch " + str(batch)
    data, labels = get_data("data_batch_" + str(batch+1))
    split_data = get_split_data(data)

    means = np.zeros((num_labels, num_dims))
    for label_i in range(num_labels):
        working_set = split_data[label_i]
        means[label_i] = np.mean(working_set, 0)
    print means

    mean_batch[batch] = means
"""
Averagin the mean for a category across all batches
"""
means = np.mean(mean_batch, 0)

"""
As shown in the book, similarity of two vectors can be compututed in either a difference or dot-product way
This is the dot-product way.
"""
W = np.zeros((num_labels,num_labels))
for i in range(num_labels), 0:
    for j in range(num_labels):
        W[i,j] = means[i].dot(means[j])

eival, eivec = np.linalg.eig(W)

"""
Getting the eigenvectors corresponding the the largest eigenvalues
"""
idx = eival.argsort()[::-1]
eivec = eivec[:,idx]

"""
Since this is a 2-D map, only 2 eigenvalues, eigenvectors will be used
"""
eival_r = np.diag(eival[0:2])**0.5
Ur_T = eivec[:,0:2].T

V_T = eival_r.dot(Ur_T)

fig, ax = plt.subplots()
ax.scatter(V_T[0], V_T[1])
plt.title("2D Map of Categories using PCoA")

for i, txt in enumerate(label_dict):
    ax.annotate(txt, (V_T[0,i], V_T[1,i]))
fig.savefig("part-b map.png")
