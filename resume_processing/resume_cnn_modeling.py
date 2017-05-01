#  Resume CNN - Author: Terry James
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

###############
##  IMPORTS  ##
###############

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint
import tensorflow as tf
import numpy as np
import threading
import warnings
import random
import time
import math
import sys
import ast
import re
import os

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

#################
##  FUNCTIONS  ##
#################

# Loading In Resume Data
def data_load(max_len):
	print('Loading data...')
	# Loading Labels
	f_labels_mapping = open('resume_embedded_mapping_labels.txt','r')
	for line in f_labels_mapping:
		labels_mapping = ast.literal_eval(line.replace('\n',''))
	labels = []
	f_labels = open('resume_embedded_labels.txt','r')
	for line in f_labels:
		row = labels_mapping[line.replace('\n','')]
		labels.append(row)
	f_labels.close()
	# Loading Data
	data = []
	f_data = open('resume_embedded_data.txt','r')
	for line in f_data:
		row = ast.literal_eval(line.replace('\n',''))
		row = row[0:max_len] if len(row) > max_len else row + [0]*(max_len-len(row))
		row = np.asarray(row)
		data.append(row)
	f_data.close()
	# Cast And Return
	labels = np.asarray(labels)
	data = np.asarray(data)
	return data, labels

# Balancing Classes Among Data Samples
def class_balance(data, labels):
	print('Balancing classes...')
	# Finding Smallest Class
	classes = {}
	for label in labels:
		if label not in classes: classes[label] = 0
		classes[label] = classes[label] + 1
	n_min_class = min(classes.values())
	# Pulling Necessary Samples
	classes = {}; new_data = []; new_labels = []
	for i in range(0,len(labels)):
		label = labels[i]
		if label not in classes: classes[label] = 0
		if classes[label] >= n_min_class: continue
		classes[label] = classes[label] + 1
		new_labels.append(label)
		new_data.append(data[i])
	# Randomizing
	data = []; labels = []
	ordering = [j for j in range(0,len(new_labels))]
	random.shuffle(ordering)
	for j in range(0,len(new_labels)):
		cur_idx = ordering[j]
		labels.append(new_labels[cur_idx])
		data.append(new_data[cur_idx])
	# Return Output
	labels = np.asarray(labels)
	data = np.asarray(data)
	return data, labels

# Grabbings Vector Embeddings For Each Vocab Word
def grab_vocab_embeddings():
	vocab_embeddings = []
	f_vocab_embeddings = open('resume_embedded_vectors.txt','r')
	for line in f_vocab_embeddings:
		embedding = np.asarray(ast.literal_eval(line.replace('\n','')))
		vocab_embeddings.append(embedding)
	vocab_embeddings = np.asarray(vocab_embeddings)
	return vocab_embeddings

# Grabbing Mapping From Words To Ids
def grab_vocab_mapping():
	f_vocab_mapping = open('resume_embedded_mapping_vocab.txt','r')
	for line in f_vocab_mapping:
		vocab_mapping = ast.literal_eval(line.replace('\n',''))
	return vocab_mapping

# Grabbing Mapping From Labels To Ids
def grab_labels_mapping():
	f_labels_mapping = open('resume_embedded_mapping_labels.txt','r')
	for line in f_labels_mapping:
		labels_mapping = ast.literal_eval(line.replace('\n',''))
	return labels_mapping

# Logging The Dimensions Of Each NN Layer
def log_layer_dims(layer_string, layer):
	print('\n' + layer_string); print(layer)
	return

# ConvNet Frame
def cnn_model_fn(features, labels, mode, params):
	"""Model function for CNN."""

	# Additional params
	text_length = params['text_length']
	vocab_size = len(params['vocab_mapping'])
	embedding_dim = len(params['vocab_embeddings'][0])
	n_classes = len(params['labels_mapping'])
	conv1_n_filters = 32
	conv1_kernel_height = 3
	conv1_u_pad = math.floor(conv1_kernel_height / 2.0)
	conv1_l_pad = math.ceil((conv1_kernel_height / 2.0) - 1)

	# Embedding Layer
	W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),trainable=False, name="W")
	W_embeddings = W.assign(params['vocab_embeddings'])
	embedded_input = tf.nn.embedding_lookup(W_embeddings, features, name="embedded_inputs")

	# Input Layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	input_layer = tf.reshape(embedded_input, [-1, text_length, embedding_dim, 1])
	log_layer_dims('input_layer', input_layer)

	# Convolutional Layer #1
	# Computes 32 features using a 5x5 filter with ReLU activation.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, text_length, embedding_dim, 1]
	# Output Tensor Shape: [batch_size, text_length, 1, 32]
	padded_input_layer = tf.pad(input_layer, [[0, 0], [conv1_u_pad, conv1_l_pad], [0, 0], [0, 0]], "CONSTANT")
	conv1 = tf.layers.conv2d(
		inputs=padded_input_layer,
		filters=conv1_n_filters,
		kernel_size=[conv1_kernel_height, embedding_dim],
		padding="valid",
		activation=tf.nn.relu)
	log_layer_dims('conv_layer', conv1)

	# Pooling Layer #1
	# First max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, text_length, 1, 32]
	# Output Tensor Shape: [batch_size, 1, 1, 32]
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[text_length, 1], strides=2)
	log_layer_dims('pool_layer', pool1)

	# Flatten tensor into a batch of vectors
	# Input Tensor Shape: [batch_size, text_length, 1, 64]
	# Output Tensor Shape: [batch_size, n_conv1_filters]
	pool1_flat = tf.reshape(pool1, [-1, conv1_n_filters])
	log_layer_dims('pool_flat_layer', pool1_flat)

	# Dense Layer
	# Densely connected layer with 1024 neurons
	# Input Tensor Shape: [batch_size, n_conv1_filters]
	# Output Tensor Shape: [batch_size, 1024]
	dense = tf.layers.dense(inputs=pool1_flat, units=1024, activation=tf.nn.relu)
	log_layer_dims('dense_layer', dense)

	# Add dropout operation; 0.6 probability that element will be kept
	dropout = tf.layers.dropout(
		inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

	# Logits layer
	# Input Tensor Shape: [batch_size, 1024]
	# Output Tensor Shape: [batch_size, n_classes]
	logits = tf.layers.dense(inputs=dropout, units=n_classes)
	log_layer_dims('logits_layer', logits)

	loss = None
	train_op = None

	# Calculate Loss (for both TRAIN and EVAL modes)
	if mode != learn.ModeKeys.INFER:
		onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=n_classes)
		loss = tf.losses.softmax_cross_entropy(
			onehot_labels=onehot_labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == learn.ModeKeys.TRAIN:
		train_op = tf.contrib.layers.optimize_loss(
			loss=loss,
			global_step=tf.contrib.framework.get_global_step(),
			learning_rate=0.001,
			optimizer="SGD")

	# Generate Predictions
	predictions = {
		"classes": tf.argmax(
		 	input=logits, axis=1),
		"probabilities": tf.nn.softmax(
			logits, name="softmax_tensor")
	}

	# Return a ModelFnOps object
	return model_fn_lib.ModelFnOps(
		mode=mode, predictions=predictions, loss=loss, train_op=train_op)

############
##  MAIN  ##
############

def main(unused_argv):

	# Loading In Resume Data
	data, labels = data_load(max_len=30)

	# Class Balance
	data, labels = class_balance(data, labels)

	# Setting training and eval data
	n_data_samples = len(data)
	train_test_split = math.floor(n_data_samples*0.7)
	train_data = data[0:train_test_split]
	train_labels = labels[0:train_test_split]
	eval_data = data[train_test_split:n_data_samples]
	eval_labels = labels[train_test_split:n_data_samples]

	# Model Parameters
	print('Loading model parameters...')
	model_params = {}
	model_params['vocab_embeddings'] = grab_vocab_embeddings()
	model_params['vocab_mapping'] = grab_vocab_mapping()
	model_params['labels_mapping'] = grab_labels_mapping()
	model_params['text_length'] = len(train_data[0])

	# Create the Estimator
	mnist_classifier = learn.Estimator(
		model_fn=cnn_model_fn, model_dir="tmp/resume_convnet_model", params=model_params)

	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	tensors_to_log = {"embedded_inputs": "embedded_inputs"} #{"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=200)

	# Train the model
	mnist_classifier.fit(
		x=train_data,
		y=train_labels,
		batch_size=100,
		steps=20000) # monitors=[logging_hook]

	# Configure the accuracy metric for evaluation
	metrics = {
		"accuracy":
			learn.MetricSpec(
				metric_fn=tf.metrics.accuracy, prediction_key="classes"),
	}

	# Evaluate the model and print results
	eval_results = mnist_classifier.evaluate(
		x=eval_data, y=eval_labels, metrics=metrics)
	print(eval_results)

if __name__ == "__main__":
	tf.app.run()
