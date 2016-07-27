#!/usr/bin/env python
#-*- encoding: utf-8 -*-
from sklearn.externals import joblib

def get_score(data_x):
	model = joblib.load('train_model.pkl')
	predict_y_prob = model.predict_proba(data_x)
	predict_y_prob_true = [y[1] for y in predict_y_prob]
	score_y = [int(y * 80000) for y in predict_y_prob_true]
	return score_y

if __name__ == '__main__':
	data_x = [0.190583172,-0.158243377,-0.360389496,-0.384792942,-0.343424735,-0.106224181,-0.113682634,-0.126039677,-0.99673324,-0.147008724,-0.130157348,-0.22143031,-0.572347219,0.100479669,-0.62220204,-0.394648284,0.187284114,0.290159813,-0.577528678,-0.166809881,18.13726469,0.441917065,0.277151756,0.351345572,-1.032848135,0.926241131,0.244116339,0.245712507,0.235608163,0.469671622,0.122129974,0.728407222,0.34102992,0.167562119,0.251952336,0.914433937,0.922824126,0.646090845]
	print get_score(data_x)
