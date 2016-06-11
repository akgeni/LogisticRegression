# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 21:48:17 2016

@author: akgeni
"""

import sframe
import string

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
def read_and_clean_data():
	'''Reads data and clean the words with punctuation
	e.g. Hello! => Hello
	'''
	products = sframe.SFrame('amazon_baby.gl/')
	products['clean_review'] = products['review'].apply(remove_punctuation)
	
	# Filter the neutral ratings, neutral ratings does not help much in
	# learning process
	products = products[products['rating'] != 3]
	
	# Add sentiment feature to indicate whether reviwe is positive or negative
	# this will be out target of prediction
	products['sentiment'] = products['rating'].apply(lambda rating : \
							1 if rating > 3 else -1)
	return products

def remove_punctuation(text):
	'''returns punctuation removed text.
	'''
	return text.translate(None, string.punctuation)


def bagof_words(data):
	'''return the a sparse matrix where each row is the word count vector
	for the corresponding review
	'''
	
	return vectorizer.fit_transform(data['clean_review'])
	
	
def transfrom_data(data):
	return vectorizer.transform(data)
	


