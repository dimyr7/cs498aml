#! env python
import numpy as np

folder_path = "./cifar-10-batches-py/"
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



batch = unpickle(folder_path + "data_batch_1")
data = np.array(batch["data"])
labels = np.array(batch["labels"])
## Data preprocessing

split_data = [0]*num_labels
for label_i in range(num_labels):
    split_data[label_i] = data[labels == label_i]


for label_i  in range(num_labels):
    working_set = split_data[label_i]
    mean_image = np.zeros(num_dims)
    for data_i in range(len(working_set)):
        mean_image += working_set[data_i]
    mean_image = mean_image/len(working_set)
    print mean_image



#def normalize((data,labels), means):
#    for i in range(num_labels):
#        if
