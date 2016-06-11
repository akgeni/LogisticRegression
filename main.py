# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 21:48:17 2016

@author: akgeni
"""
import prepareData
import learnData

if __name__ == '__main__':
	products = prepareData.read_and_clean_data();
	
	# split the data into training and test set.
	train_data, test_data = products.random_split(0.8, seed=1)
	# get the word frequency for each review
	train_matrix = prepareData.bagof_words(train_data['clean_review'])
	test_matrix = prepareData.bagof_words(test_data['clean_review'])
	
	# train the logistic classifier
	sentiment_model = learnData.logistic_classifier(train_matrix, train_data['sentiment'])
	
