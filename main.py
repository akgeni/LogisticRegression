# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 21:48:17 2016

@author: akgeni@gmail.com
"""
import sys

import prepareData
import learnData

from learnData import map_prediction
if __name__ == '__main__':
	products = prepareData.read_and_clean_data();
	
	print("data is ready for training")
	print(products.head(5))
	
	# sample data to 30% so that quickly see the output.
	# comment it for complete data set.
	products = products.sample(0.3)
	
	# split the data into training and test set.
	train_data, test_data = products.random_split(0.8, seed=1)
	
	# get the word frequency for each review
	train_matrix = prepareData.bagof_words(train_data)
	test_matrix = prepareData.transfrom_data(test_data['clean_review'])
	print("Got bag of words")
	
	# train the logistic classifier
	sentiment_model = learnData.logistic_classifier(train_matrix, train_data['sentiment'])
	
	# lets test some review sentiments
	sample_test_data = test_data[12:14]
	print(sample_test_data[0]['clean_review'])
	print("")
	print(sample_test_data[1]['clean_review'])
	
	
	# to predict above reviews first we need to transform these review into
	# bag of important words
	sample_test_matrix = prepareData.transfrom_data(sample_test_data['clean_review'])
	
	predictions = sentiment_model.decision_function(sample_test_matrix)
	for data, pred in zip(sample_test_data, predictions):
		print(data['clean_review'], "=> ", map_prediction(pred))
	
	
