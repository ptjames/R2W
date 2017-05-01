###############
##  IMPORTS  ##
###############

from pprint import pprint
import numpy as np
import threading
import warnings
import random
import gensim
import time
import sys
import ast
import re
import os

#################
##  FUNCTIONS  ##
#################

# Loading In Resume Data
def data_load(data_file):
	data = {}
	print('loading data from: ' + data_file)
	# Iterate Over File Lines
	f = open(data_file,'r')
	for line in f:
		row = line.replace('\n','')
		# Pull File Data Into Dictionary
		if len(row) > 5:
			row = ast.literal_eval(row)
			key = row['resume_link']
			data[key] = row
	# Close File And Return
	f.close()
	return data

# Formatting Resume Data
def format_data(data):
	# Load Word2Vec Model
	wk_dir = os.getcwd()
	f_name = wk_dir.replace('resume_processing','nlp_opensource_models\GoogleNews-vectors-negative300.bin')
	model = gensim.models.KeyedVectors.load_word2vec_format(f_name, binary=True)
	# Iterate Over Resumes
	unique_resumes = {}; labels = {}; vocabulary = {}; embeddings = {}; data_samples_counter = 0
	for resume in data:
		# Check For Duplicate Resumes
		if resume in unique_resumes:
			continue
		# Iterate Over Resume Sections (Each Data Sample = Resume Section)
		for resume_elem in data[resume]:
			cur_elem = data[resume][resume_elem]
			# Format Data Labels
			if resume_elem[0:3] == 'job':
				data_label = 'job'
			elif resume_elem[0:3] == 'edu':
				data_label = 'edu'
			else:
				continue
			if data_label not in labels:
				labels[data_label] = len(labels)
			# Format Data Features
			data_elems = [cur_elem[x] for x in cur_elem if len(cur_elem[x]) > 2]
			data_string = '. '.join(data_elems)
			data_string = re.sub('[^a-zA-Z0-9 ]',' ',data_string)
			data_string = re.sub('[ ]{2,}',' ',data_string).lower()
			data_string_elems = data_string.split(' ')
			data_string_embedded = []
			for word in data_string_elems:
				try:
					word_embedded = list(model[word])
					if word not in vocabulary:
						word_id = len(vocabulary)
						vocabulary[word] = word_id
						embeddings[word_id] = word_embedded
					data_string_embedded.append(vocabulary[word])
				except:
					continue
			if len(data_string_embedded) == 0:
				continue
			# Output Formatted Data
			output_data(data_string_embedded, data_label, data_samples_counter)
			data_samples_counter = data_samples_counter + 1
		# For Duplicate Checking
		unique_resumes[resume] = 1
	# Output Labels And Vocabulary Mapping
	f_out_labels = open('resume_embedded_mapping_labels.txt','w')
	output = str(labels)
	f_out_labels.write(output)
	f_out_labels.close()
	f_out_vocab = open('resume_embedded_mapping_vocab.txt','w')
	output = str(vocabulary)
	f_out_vocab.write(output)
	f_out_vocab.close()
	f_out_embeddings = open('resume_embedded_vectors.txt','w')
	len_embeddings = len(embeddings)
	for i in range(0,len_embeddings):
		suffix = '\n' if i != (len_embeddings - 1) else ''
		output = str(embeddings[i]) + suffix
		f_out_embeddings.write(output)
	f_out_embeddings.close()
	# Return
	return

# Output Formatted Data
def output_data(formatted_data, formatted_label, data_samples_counter):
	output_mode = 'w' if data_samples_counter == 0 else 'a'
	# Outputting Labels
	f_out_labels = open('resume_embedded_labels.txt',output_mode)
	output = str(formatted_label) + '\n'
	f_out_labels.write(output)
	f_out_labels.close()
	# Outputting Data
	f_out_data = open('resume_embedded_data.txt',output_mode)
	output = str(formatted_data) + '\n'
	f_out_data.write(output)
	f_out_data.close()
	return

############
##  MAIN  ##
############

# Loading In Resume Data
data_file = 'resume_scraping_output_small.txt'
data = data_load(data_file)

# Formatting Resume Data
format_data(data)
