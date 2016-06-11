# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 21:48:17 2016

@author: akgeni
"""
from sklearn.linear_model import LogisticRegression
import sklearn

def logistic_classifier(data, target):
	''' data is sparse representation, while target is sentiment
	'''
	sentiment_model = sklearn.linear_model.LogisticRegression()
	sentiment_model.fit(data, target)
	print("You logistic classifier is ready")
	return sentiment_model

def map_prediction(score):
	return "Positive Review" if score > 0 else "Negative Review"
