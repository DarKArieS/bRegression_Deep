# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:03:54 2018

@author: Aries
"""

import pandas 
import matplotlib.pyplot as plt

def load_plot_Fit_History(historyfiles, model_label, column, outfilename):
	fig = plt.figure(figsize=(8,6))
	for x,i in enumerate(historyfiles):
		dataframe = pandas.read_csv(i, delim_whitespace=True, header=0)
		d = dataframe.values[:,column]
		plt.plot(d,label = model_label[x])
	plt.legend(loc='best')
	fig.savefig(str(outfilename)+'.png')

his=['History_KERAS_18var_4Dummies_cutSeries_GenOverReco_Res_model_v1_ep150_batch512_SklearnScaled_2xdata.txt',
     'History_KERAS_18var_4Dummies_cutSeries_GenOverReco_Res_model_v2_ep100_batch512_SklearnScaled.txt'
     ]
labell=['Res_v1_ep150','Res_v2_ep100']
load_plot_Fit_History(his, labell,1,'testtt.png')

