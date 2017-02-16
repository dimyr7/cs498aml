#! env python
import numpy as np

folder_path = "./cifar-10-batches-py/"
num_labels = 10
num_dims = 3072
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

batch = unpickle(folder_path + "data_batch_1")
data = np.array(batch["data"])
label = np.array(batch["labels"])

num_data = len(label1)

print data1.shape
print label1.shape

mean_images = np.zeros((num_labels, num_dims))
for label_i in range(num_labels):
    number_occurences = 0
    for data_i in range(num_data):
        if(label1[data_i] != label_i):
            continue
        number_occurences += 1
        xi = data1[data_i]
        mean_images[label_i] += xi
    mean_images[label_i] = mean_images[label_i]/number_occurences


def normalize(data, means):
    for i in range(num_labels):

