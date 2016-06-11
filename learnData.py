# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 21:48:17 2016

@author: akgeni
"""
from sklearn.linear_model import LogisticRegression

def logistic_classifier(data, target):
	''' data is sparse representation, while target is sentiment
	'''
	model = LogisticRegression()
	model.fit(data, target)
	
	return model
