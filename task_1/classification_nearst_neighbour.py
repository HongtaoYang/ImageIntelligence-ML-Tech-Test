import numpy as np
import pickle
import os
import glob
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def read_data(data_path):
	all_samples = []
	all_labels = []

	all_subjs = os.listdir(data_path)
	for n, subj in enumerate(all_subjs):
		im_feats = glob.glob(os.path.join(data_path, subj, '*.npy'))
		num_feats = len(im_feats)
		label = [n] * num_feats
		all_samples.extend(im_feats)
		all_labels.extend(label)

	feat_train, feat_test, l_train, l_test = train_test_split(all_samples, all_labels, test_size=0.2)

	# read numpy array into memory
	feat_train_arr = np.empty([len(feat_train), 512], dtype=np.float32)
	feat_test_arr = np.empty([len(feat_test), 512], dtype=np.float32)
	for i, np_file in enumerate(feat_train):
		feat_train_arr[i, :] = np.load(os.path.join(data_root, np_file))
	for i, np_file in enumerate(feat_test):
		feat_test_arr[i, :] = np.load(os.path.join(data_root, np_file))


	return feat_train_arr, feat_test_arr, np.array(l_train), np.array(l_test)


data_root = '/home/hongtao/my_projects/ImageIntelligence ML Tech Test/faces'
feat_train_arr, feat_test_arr, label_train, label_test = read_data(data_root)

# for each test sample, find the most similar one.
sim_mat = distance.cdist(feat_test_arr, feat_train_arr, 'sqeuclidean')
print sim_mat.shape

# find the nearst neighbour and calculate accuracy
nearst = np.argmin(sim_mat, axis=1)
prediction = [label_train[nearst[i]] for i in range(len(label_test))]

acc = accuracy_score(label_test, prediction)
print acc