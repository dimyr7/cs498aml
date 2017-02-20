#! env python
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

folder_path = "./cifar-10-batches-py/"
num_pcas = 20

def unpickle(file):
    """
    Given the dataset that was provided, cPickle transforms it into a dictionary
    of data, labels, etc...
    """
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

# extract the meta data and store them in a variable to use later
meta_info = unpickle(folder_path + "batches.meta")
label_dict = meta_info["label_names"]
num_labels = len(label_dict)
num_dims = meta_info["num_vis"]
num_data = meta_info["num_cases_per_batch"]

def show_pic(data, name):
    """
    performs a plot given the data and the name of the image

    transforms the data array into a format the matplotlib accepts
    """
    pic = np.zeros((32,32,3))
    for i in range(32):
        for j in range(32):
            for k in range(3):
                pic[i,j,k] = data[1024*k + 32*j + i]/256.
    plt.imsave(fname=name, arr=pic)

def get_split_data(data):
    """
    returns an 10 matrices, where each matrix corresponds to the data of a label
    """
    split_data = [0]*num_labels
    for label_i in range(num_labels):
        split_data[label_i] = data[labels == label_i]
    return split_data

def get_data(batch_file):
    """
    extracts the data and label from the batch file
    """
    batch = unpickle(folder_path + str(batch_file))
    data = np.array(batch["data"], dtype=np.float32)[0:10000]
    labels = np.array(batch["labels"])[0:10000]
    return data, labels

def get_centered_working_set(working_set, label_index):
    """
    given a working set, return the centered working set

    This function also plots the mean of the working set
    """
    mean_i = np.mean(working_set, 0)
    show_pic(mean_i, "mean" + str(label_index) + ".png")      # plot the mean
    centered = working_set - mean_i
    return centered

def get_approx_data(working_set, eivec, label_index):
    """
    returns the first 20 principal components which are the summary of the working set

    It does so by taking the dot product of the eigen vectors and each image in the working set.
    The function then returns the first 20 columns, principal components
    """
    rotated_working_set = np.zeros(centered_working_set.shape)
    for data_i in range(len(working_set)):
        rotated_working_set[data_i] = np.dot(eivec.T, working_set[data_i])
    return rotated_working_set[:, 0:num_pcas], approx_data

def get_sorted_eigvec(working_set):
    """
    given the working set return the eigen vectors sorted by the corresponding eigen values
    """
    cov_mat = np.cov(working_set.T)
    eival, eivec = np.linalg.eig(cov_mat)

    eig_idx = eival.argsort()[::-1]
    eivec = eivec[:, eig_idx]

    return eivec

num_batches = 5
error = [0]*num_labels

# run the code for all the batches
for batch in range(1, num_batches + 1):
    data, labels = get_data("data_batch_" + str(batch))
    split_data = get_split_data(data)                            # get the splitted data

    # calculate the error for each class
    for label_i in range(num_labels):
        print "working on " + label_dict[label_i]
        working_set = split_data[label_i]
        N = len(working_set)

        centered_working_set = get_centered_working_set(working_set, label_i)       # get the centered working set
        eigen_vec = get_sorted_eigvec(centered_working_set)                         # get the sorted eigen vectors
        # get the first 20 principal components
        rotated_working_set , approx_data = get_approx_data(centered_working_set, eigen_vec, label_i)

        error_sum = 0.
        for data_i in range(N):
            the_diff = (rotated_working_set[data_i] - approx_data[data_i])
            error_diff = (np.linalg.norm(the_diff)/np.linalg.norm(rotated_working_set[data_i]))**2
            error_sum += error_diff
        error[label_i] += (error_sum/N)


# take the average error over all the batches
error /= num_batches

# Plot the error for all the classes
plt.figure()
plt.title("Errors as a function of category")
categories = tuple(label_dict)
plt.bar(np.arange(len(categories)), error, align='center')
y_pos = np.arange(len(categories))
plt.xlabel("Label")
plt.xticks(y_pos, categories, rotation=20)
plt.ylabel("Error")
plt.savefig("part-a errors.png")
