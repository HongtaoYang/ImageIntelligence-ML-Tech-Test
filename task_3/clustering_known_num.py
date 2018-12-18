import numpy as np
import pickle
import os
import glob
import random
import time
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.linear_assignment_ import linear_assignment
import matplotlib.pyplot as plt


def clustering_acc(y_true, y_pred):
	y_true = y_true.astype(np.int64)
	assert y_pred.size == y_true.size
	D = max(y_pred.max(), y_true.max()) + 1
	w = np.zeros((D, D), dtype=np.int64)
	for i in range(y_pred.size):
		w[y_pred[i], y_true[i]] += 1
	ind = linear_assignment(w.max() - w)
	
	return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def read_data(data_path, samples_per_person):
	all_samples = []
	all_labels = []

	all_subjs = os.listdir(data_path)
	for n, subj in enumerate(all_subjs):
		im_feats = glob.glob(os.path.join(data_path, subj, '*.npy'))
		num_feats = len(im_feats)
		if samples_per_person is not None:
			if samples_per_person <= num_feats:
				selected_feats = random.sample(im_feats, samples_per_person)
				label = [n] * samples_per_person
			else:
				selected_feats = im_feats
				label = [n] * num_feats

			all_samples.extend(selected_feats)
			all_labels.extend(label)

		else:
			label = [n] * num_feats
			all_samples.extend(im_feats)
			all_labels.extend(label)


	# read numpy array into memory
	feat_arr = np.empty([len(all_samples), 512], dtype=np.float32)
	for i, np_file in enumerate(all_samples):
		feat_arr[i, :] = np.load(os.path.join(data_root, np_file))


	return feat_arr, np.array(all_labels)


data_root = '/home/hongtao/my_projects/ImageIntelligence ML Tech Test/faces'


all_feat_arr, all_labels = read_data(data_root, samples_per_person=None)
start = time.time()
sim_mat = distance.cdist(all_feat_arr, all_feat_arr, 'sqeuclidean')
agglo = AgglomerativeClustering(n_clusters=158, affinity='precomputed', linkage='average')
cluster = agglo.fit(sim_mat).labels_
end = time.time()
acc = clustering_acc(np.array(all_labels), cluster)
print end-start
print acc


# acc_record = []
# # time_record = []
# for num_imgs in range(5, 540, 10):
# 	print num_imgs
# 	all_feat_arr, all_labels = read_data(data_root, samples_per_person=num_imgs)

# 	# compute the affinity matrix based on the squared Euclidean distance
# 	# start = time.time()
# 	sim_mat = distance.cdist(all_feat_arr, all_feat_arr, 'sqeuclidean')
# 	agglo = AgglomerativeClustering(n_clusters=158, affinity='precomputed', linkage='average')
# 	cluster = agglo.fit(sim_mat).labels_
# 	# end = time.time()

# 	# tiem_elapsed = end-start
# 	acc = clustering_acc(np.array(all_labels), cluster)
# 	acc_record.append(acc)
# 	# time_record.append(tiem_elapsed)

# # plot the results
# plt.plot(range(5, 540, 10), acc_record)
# plt.ylabel('clustering accuracy')
# plt.xlabel('imgs per person')
# plt.show()



# # plot the results
# x = range(5, 540, 10)
# y1 = acc_record
# y2 = time_record

# fig, ax1 = plt.subplots()
# ax1.set_xlabel('imgs per person')
# ax1.set_ylabel('clustering accuracy', color='tab:blue')
# ax1.plot(x, y1, color='tab:blue')
# ax1.tick_params(axis='y', labelcolor='tab:blue')

# ax2 = ax1.twinx()
# ax2.set_ylabel('running time (s)', color='tab:red')
# ax2.plot(x, y2, color='tab:red')
# ax2.tick_params(axis='y', labelcolor='tab:red')

# plt.xlabel('imgs per person')
# plt.ylabel('time')

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
