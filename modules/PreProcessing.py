import pandas
import numpy as np
from sklearn.utils import shuffle

def series_cut(dataframe_):
	#remove extreme value, return dataframe
	dataframe = dataframe_.copy()
	Cut_Value = [
	['target',0.4,1.6],
	['Jet_genjetPt',0,500],
	['Jet_pt',0,500],
	['Jet_mt',0,500],
	['Jet_leadTrackPt',0,200],
	['Jet_leptonPtRel_new',0,5],
	['Jet_leptonPt',0,200],
	['Jet_vtxPt',0,200],
	['Jet_vtx3dL',0,6],
	['Jet_vtx3deL_new',0,0.5],
	['Jet_PFMET',0,300]
	]
	
	print('dataframe:' + str(dataframe.shape))

	for nCol in range(len(Cut_Value)):
		print('selection for column: ' + dataframe.columns[nCol])
		ColumnName=Cut_Value[nCol][0]
		dataframe = dataframe.drop(dataframe.get(ColumnName)[(dataframe.get(ColumnName)<Cut_Value[nCol][1])].index)
		dataframe = dataframe.drop(dataframe.get(ColumnName)[(dataframe.get(ColumnName)>Cut_Value[nCol][2])].index)
		print(dataframe.shape)
	
	return dataframe
	
def target_cut(dataframe_):
	#remove extreme value, return dataframe
	dataframe = dataframe_.copy()
	Cut_Value = [['target',0.4,1.6]]
	
	print('dataframe:' + str(dataframe.shape))

	for nCol in range(len(Cut_Value)):
		print('selection for column: ' + dataframe.columns[nCol])
		ColumnName=Cut_Value[nCol][0]
		dataframe = dataframe.drop(dataframe.get(ColumnName)[(dataframe.get(ColumnName)<Cut_Value[nCol][1])].index)
		dataframe = dataframe.drop(dataframe.get(ColumnName)[(dataframe.get(ColumnName)>Cut_Value[nCol][2])].index)
		print(dataframe.shape)
	
	return dataframe
	
def simple_scale(dataframe, Col_start = 0, Col_stop = None):
	# scale the data between 0 and 1
	# only worked in python3
	# return dataset
	from mlxtend.preprocessing import minmax_scaling
	dataset = dataframe.values.copy()
	
	Col_stop_ = len(dataframe.columns)
	if Col_stop is not None: Col_stop_ = Col_stop
	
	for nCol in range(Col_start,Col_stop_):
		dataset[:,nCol] = minmax_scaling(dataset,columns = [nCol])[:,0]
	
	return dataset
	
def simple_scale_dataset(dataset, Col_start = 0, Col_stop = None):
	# scale the data between 0 and 1
	# only worked in python3
	# return dataset
	from mlxtend.preprocessing import minmax_scaling
	
	Col_stop_ = dataset.shape[1]
	if Col_stop is not None: Col_stop_ = Col_stop
	
	for nCol in range(Col_start,Col_stop_):
		dataset[:,nCol] = minmax_scaling(dataset,columns = [nCol])[:,0]
	
	print('simple scale data to [0,1]')
	print('Mean:'+str(dataset.mean(axis=0)))
	print('Std:'+str(dataset.std(axis=0)))
	
	return dataset
	
def boxcox_normalized(dataframe, Col_start = 0, Col_stop = None):
	Col_stop_ = len(dataframe.columns)
	if Col_stop is not None: Col_stop_ = Col_stop

	# should remove zero value and negative value
	from scipy import stats
	
	# all positive, remove zero
	for nCol in range(Col_start,Col_stop_):
		ColumnName=dataframe.columns[nCol]
		dataframe.get(ColumnName)[dataframe.get(ColumnName)<0] *= -1

		
	dataset = dataframe.values.copy()
	
	for nCol in range(Col_start,Col_stop_):
		print('Deal with: ' + str(dataframe.columns[nCol]))
		dataset[:,nCol] = stats.boxcox(dataset[:,nCol])[0]
		
	return dataset

def sklearn_simple_scale(dataset):
	#sklearn preprocessing
	# mean=0, std=1
	# better for Initializer glorot_uniform?
	
	from sklearn import preprocessing
	# dataset = dataset_.copy()
	dataset = preprocessing.scale(dataset)
	
	print('scale data with Mean=0, Std=1')
	print('Mean:'+str(dataset.mean(axis=0)))
	print('Std:'+str(dataset.std(axis=0)))
	return dataset

def read_dataset(nrows_ = None, skiprows_ = None):
	pandas.options.mode.use_inf_as_na = True
	dataframe = pandas.read_csv("../Data/training/minitree_4b_180710.txt", delim_whitespace=True, header=0, skiprows=skiprows_, nrows=nrows_)
	dataframe = shuffle(dataframe)
	dataset_tmp = dataframe.values.copy()
	target = dataset_tmp[:,0]/dataset_tmp[:,1]
	dataframe.insert(0,'target',target)
	#====================data cleaning=====================
	# ● check no vertex
	# Get no vertex events index
	NoVtx = dataframe.loc[dataframe['Jet_vtx3deL_new'].isnull()].index

	# ● check no lepton track
	# Get no vertex events index
	NoLepton = dataframe.loc[dataframe['Jet_leptonPtRel_new'].isnull()].index

	# ● check no charged tracks (very rare)
	NoChTrk = dataframe.loc[dataframe.Jet_leadTrackPt == -99].index

	# ● Lead Track = Lepton
	HasLeadLepton = dataframe.loc[dataframe.Jet_leadTrackPt == dataframe.Jet_leptonPt].index

	# dummy variables for vertex/track/lepton
	dataframe['HasVertex'] = pandas.Series(1, index=dataframe.index)
	dataframe.loc[NoVtx,'HasVertex'] = 0
	dataframe.loc[NoVtx,'Jet_vtx3deL_new'] = 0
	dataframe[['HasVertex','Jet_vtx3deL_new']]

	dataframe['HasLepton'] = pandas.Series(1, index=dataframe.index)
	dataframe.loc[NoLepton,'HasLepton'] = 0
	dataframe.loc[NoLepton,'Jet_leptonPtRel_new'] = 0
	dataframe.loc[NoLepton,'Jet_leptonPt'] = 0
	dataframe.loc[NoLepton,'Jet_leptonDeltaR'] = 0
	dataframe[['HasLepton','Jet_leptonPtRel_new','Jet_leptonPt']]

	dataframe['HasChTrk'] = pandas.Series(1, index=dataframe.index)
	dataframe.loc[NoChTrk,'HasChTrk'] = 0
	dataframe.loc[NoChTrk,'Jet_leadTrackPt'] = 0
	dataframe.ix[dataframe.loc[dataframe.Jet_leadTrackPt == -99].index,['HasChTrk','Jet_leptonPt','Jet_leadTrackPt']]

	dataframe['HasLeadLepton'] = pandas.Series(0, index=dataframe.index)
	dataframe.loc[HasLeadLepton,'HasLeadLepton'] = 1
	
	return series_cut(dataframe)
	# return target_cut(dataframe)
	# return dataframe
	
'''
def read_test_dataset(nrows_ = None, skiprows_ = None):
	pandas.options.mode.use_inf_as_na = True
	dataframe = pandas.read_csv("../Data/testing/minitree_13TeV_G_m250_dijet_noreg.txt", delim_whitespace=True, header=0, skiprows=skiprows_, nrows=nrows_)
	dataset_tmp = dataframe.values.copy()
	target = dataset_tmp[:,0]/dataset_tmp[:,1]
	dataframe.insert(0,'target',target)
	#====================data cleaning=====================
	# ● check no vertex
	# Get no vertex events index
	NoVtx = dataframe.loc[dataframe['Jet_vtx3deL_new'].isnull()].index

	# ● check no lepton track
	# Get no vertex events index
	NoLepton = dataframe.loc[dataframe['Jet_leptonPtRel_new'].isnull()].index

	# ● check no charged tracks (very rare)
	NoChTrk = dataframe.loc[dataframe.Jet_leadTrackPt == -99].index

	# ● Lead Track = Lepton
	HasLeadLepton = dataframe.loc[dataframe.Jet_leadTrackPt == dataframe.Jet_leptonPt].index

	# dummy variables for vertex/track/lepton
	dataframe['HasVertex'] = pandas.Series(1, index=dataframe.index)
	dataframe.loc[NoVtx,'HasVertex'] = 0
	dataframe.loc[NoVtx,'Jet_vtx3deL_new'] = 0
	dataframe[['HasVertex','Jet_vtx3deL_new']]

	dataframe['HasLepton'] = pandas.Series(1, index=dataframe.index)
	dataframe.loc[NoLepton,'HasLepton'] = 0
	dataframe.loc[NoLepton,'Jet_leptonPtRel_new'] = 0
	dataframe.loc[NoLepton,'Jet_leptonPt'] = 0
	dataframe.loc[NoLepton,'Jet_leptonDeltaR'] = 0
	dataframe[['HasLepton','Jet_leptonPtRel_new','Jet_leptonPt']]

	dataframe['HasChTrk'] = pandas.Series(1, index=dataframe.index)
	dataframe.loc[NoChTrk,'HasChTrk'] = 0
	dataframe.loc[NoChTrk,'Jet_leadTrackPt'] = 0
	dataframe.ix[dataframe.loc[dataframe.Jet_leadTrackPt == -99].index,['HasChTrk','Jet_leptonPt','Jet_leadTrackPt']]

	dataframe['HasLeadLepton'] = pandas.Series(0, index=dataframe.index)
	dataframe.loc[HasLeadLepton,'HasLeadLepton'] = 1
	
	return series_cut(dataframe)
	# return target_cut(dataframe)
	# return dataframe
'''