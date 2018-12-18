import numpy as np
import pickle
import os
import glob
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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
print feat_test_arr.shape

# construct a simple fully connected neural net
num_class = 158
batch_size = 128

feat_in = tf.placeholder(shape=[None, 512], dtype=tf.float32, name='input_feature')
label_in = tf.placeholder(shape=[None], dtype=tf.int32, name='input_label')
label_in_onehot = tf.one_hot(label_in, num_class)

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(0.0001, global_step, 2000, 0.5, staircase=True)

fc1 = tf.layers.dense(feat_in, 512, activation=tf.nn.relu, name='fc1')
logits = tf.layers.dense(fc1, num_class, activation=None, name='predict')
loss = tf.losses.softmax_cross_entropy(onehot_labels=label_in_onehot, logits=logits)

optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(loss, global_step=global_step)

pred_prob = tf.nn.softmax(logits)
pred = tf.argmax(pred_prob, axis=1)


# train the network
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(10000):
		random_index = random.sample(range(len(label_train)), batch_size)
		batch_data_in = feat_train_arr[random_index, :]
		batch_label_in = np.array(label_train)[random_index]
		feed_dict = {feat_in: batch_data_in,
					 label_in: batch_label_in}

		sess.run(train_op, feed_dict=feed_dict)

		if i % 500 == 0:
			feed_dict = {feat_in: feat_test_arr}
			prediction = sess.run(pred, feed_dict=feed_dict)
			accuracy = accuracy_score(label_test, prediction)
			print('testing accuracy at %d is %f' % (i, accuracy))

