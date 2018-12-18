import numpy as np
import pickle
import os
import glob
import random
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def read_data(data_path):
	all_samples_known = []
	all_labels_known = []
	all_samples_unknown = []
	all_labels_unknown = []

	all_subjs = os.listdir(data_path)
	random.shuffle(all_subjs)
	known_person = all_subjs[0:130]
	unknown_person = all_subjs[130:]

	for n, subj in enumerate(known_person):
		im_feats = glob.glob(os.path.join(data_path, subj, '*.npy'))
		num_feats = len(im_feats)
		label = [n] * num_feats
		all_samples_known.extend(im_feats)
		all_labels_known.extend(label)
	feat_train_known, feat_test_known, l_train_known, l_test_known = train_test_split(all_samples_known, all_labels_known, test_size=0.2)

	for subj in unknown_person:
		im_feats = glob.glob(os.path.join(data_path, subj, '*.npy'))
		num_feats = len(im_feats)
		label = [9999] * num_feats
		all_samples_unknown.extend(im_feats)
		all_labels_unknown.extend(label)
	feat_train_unknown, feat_test_unknown, l_train_unknown, l_test_unknown = train_test_split(all_samples_unknown, all_labels_unknown, test_size=0.2)

	feat_train = feat_train_known + feat_train_unknown
	feat_test = feat_test_known + feat_test_unknown
	l_train = l_train_known + l_train_unknown
	l_test = l_test_known + l_test_unknown

	# read numpy array into memory
	feat_train_arr = np.empty([len(feat_train), 512], dtype=np.float32)
	feat_test_arr = np.empty([len(feat_test), 512], dtype=np.float32)
	for i, np_file in enumerate(feat_train):
		feat_train_arr[i, :] = np.load(os.path.join(data_root, np_file))
	for i, np_file in enumerate(feat_test):
		feat_test_arr[i, :] = np.load(os.path.join(data_root, np_file))

	return feat_train_arr, feat_test_arr, np.array(l_train), np.array(l_test)


def cluster_center(samples, labels):
	unique_labels = set(labels)
	print len(unique_labels)

	centers = np.empty([len(unique_labels), 512], dtype=np.float32)
	for i in unique_labels:
		index = [k for k in range(len(labels)) if labels[k] == i]
		same_class_samples = samples[index, :]
		center = np.mean(same_class_samples, axis=0)
		centers[i, :] = center

	return centers


data_root = '/home/hongtao/my_projects/ImageIntelligence ML Tech Test/faces'
feat_train_arr, feat_test_arr, label_train, label_test = read_data(data_root)

# construct clusters using known person only
known_person_index = [i for i in range(len(label_train)) if label_train[i] != 9999]
known_person_arr = feat_train_arr[known_person_index, :]
known_person_label = label_train[known_person_index]
centers = cluster_center(known_person_arr, known_person_label)

center_dist = distance.cdist(centers, centers, 'sqeuclidean')

# get max variance across all cluster, use it as the threshold
thresholds = np.zeros([len(centers)], dtype=np.float32)
for i, samples in enumerate(known_person_arr):
	d = distance.sqeuclidean(samples, centers[known_person_label[i]])
	thresholds[known_person_label[i]] = max(thresholds[known_person_label[i]], d)

# compute similarity matrix
sim_mat_train = distance.cdist(feat_train_arr, centers, 'sqeuclidean')
min_dist_train = np.min(sim_mat_train, axis=1)

# find the best threshold using trining data
accuracy = []
limits = np.arange(0, np.max(thresholds), 0.05)
for thres in limits:
	prediction = np.argmin(sim_mat_train, axis=1)
	prediction[min_dist_train > thres] = 9999
	acc = accuracy_score(label_train, prediction)
	accuracy.append(acc)

best_thres = limits[np.argmax(accuracy)]
print('threshold is %f.' % best_thres)


# run testing
sim_mat = distance.cdist(feat_test_arr, centers, 'sqeuclidean')
min_dist = np.min(sim_mat, axis=1)

prediction = np.argmin(sim_mat, axis=1)
prediction[min_dist > best_thres] = 9999
acc = accuracy_score(label_test, prediction)
print acc
