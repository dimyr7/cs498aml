#! env python
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
np.set_printoptions(threshold=np.nan)
folder_path = "./cifar-10-batches-py/"
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


meta_info = unpickle(folder_path + "batches.meta")
label_dict = meta_info["label_names"]
num_labels = len(label_dict)
num_dims = meta_info["num_vis"]
num_data = meta_info["num_cases_per_batch"]



batch = unpickle(folder_path + "data_batch_1")
data = np.array(batch["data"], dtype=np.float32)
labels = np.array(batch["labels"])
## Data preprocessing
split_data = [0]*num_labels
for label_i in range(num_labels):
    split_data[label_i] = data[labels == label_i]


for label_i  in range(num_labels):
    working_set = split_data[label_i]
    mean_image = np.zeros(num_dims)
    print(len(working_set))
    for data_i in range(len(working_set)):
        mean_image += working_set[data_i]
    mean_image = mean_image/len(working_set)
    print mean_image

#print (mean_image/len(working_set))
for k  in range(len(mean_image)):
    if(mean_image[k] > 255):
        print k
        print mean_image[k]
        print "\n"
show_pic(mean_image, "mean" + str(label_i))

# calculate cov mat
cov_mat = []
for i in range(len(split_data)):
    d = split_data[i].T
    c = np.cov(d)
    assert c.shape == (num_dims, num_dims)
    cov_mat.append(c)

#def normalize((data,labels), means):
#    for i in range(num_labels):
#        if
