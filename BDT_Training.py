# parameters
data_size = None
Variable_Type = 'KERAS_18var_4Dummies_cutSeries_GenOverReco_'
Model_name = 'XGBoost_BDT_v0'
Input_dim = '22'
# Other_cmd = '_SimpleScaled'
Other_cmd = '_SklearnScaled_fulldata'
split_validate_percent = 0.1

# #############################################################################
# Load data
# #############################################################################
import pandas
import numpy as np
import modules.PreProcessing as PreProcessing

pre_dataframe = PreProcessing.read_dataset(data_size)

#18_4dum_series_cut
dataframe = pre_dataframe[['Jet_genjetPt','Jet_pt','Jet_eta','Jet_mt','Jet_leadTrackPt','Jet_leptonPtRel_new','Jet_leptonPt','Jet_leptonDeltaR','Jet_neHEF','Jet_neEmEF','Jet_vtxPt','Jet_vtxMass','Jet_vtx3dL','Jet_vtxNtrk','Jet_vtx3deL_new','nGoodPVs','Jet_PFMET','Jet_METDPhi','Jet_JetDR','Jet_type','HasVertex', 'HasLepton', 'HasChTrk']]
dataset = dataframe.values.copy()
data = dataset[:,1:23]
target = dataset[:,0]/dataset[:,1]

#scale data
scaledata = PreProcessing.sklearn_simple_scale(data)
target = np.log1p(target)

split = int(scaledata.shape[0]*(1-split_validate_percent))
#training dataset
training_dataset = scaledata[0:split,:]
training_target = target[0:split]

#validate dataset
validate_dataset = scaledata[split:,:]
validate_target = target[split:]
# #############################################################################
import xgboost as xgb

print('train with sk API')
xgb_model = xgb.XGBRegressor(objective='reg:linear', max_depth=5, learning_rate=0.1, n_estimators=500, booster='gbtree', base_score=1).fit(training_dataset, training_target)

from sklearn.metrics import mean_squared_error
predictions = xgb_model.predict(training_dataset)
val_predictions = xgb_model.predict(validate_dataset)

print('scaled target test:')
print('Train Accuracy(MSE):', mean_squared_error(training_target, predictions))
print('Validation Accuracy(MSE):', mean_squared_error(validate_target, val_predictions))

print('target test:')
target_ori = dataset[:,0]/dataset[:,1]
training_target_ori = target_ori[0:split]
validate_target_ori = target_ori[split:]
predictions_ori = np.expm1(predictions)
val_predictions_ori = np.expm1(val_predictions)
print('Train Accuracy(MSE):', mean_squared_error(training_target_ori, predictions))
print('Validation Accuracy(MSE):', mean_squared_error(validate_target_ori, val_predictions))

import pickle
pickle.dump(xgb_model, open("XGB_addDummy_cutSeries_TargetGenPt.pkl", "wb"))

	
	

# Load_xgb_model = pickle.load(open("XGBtest.pkl", "rb"))
# Load_predictions = Load_xgb_model.predict(data)
# print(mean_squared_error(target, predictions))


# #############################################################################
# print('train with xgb API, something is wrong :(')
# T_train_xgb = xgb.DMatrix(data, target)
# params = {
# "objective": "reg:linear", "booster":"gbtree", "max_depth":"5", "learning_rate":"0.1","n_estimators":"500", "nthread":"8",
# "silent":"True", "n_jobs":"1", "gamma":"0", "min_child_weight":"1", "max_delta_step":"0", "subsample":"1", "colsample_bytree":"1", "colsample_bylevel":"1", "reg_alpha":"0", "reg_lambda":"1", "scale_pos_weight":"1", "base_score":"1", "random_state":"0"
# }
# gbm = xgb.train(dtrain=T_train_xgb,params=params)
# Y_pred = gbm.predict(xgb.DMatrix(data))
# print(mean_squared_error(target, Y_pred))
# print(Y_pred)

# #############################################################################
# import modules.Plot as Plot

# dec = []
# dec += [target]
# dec += [predictions]
# dec += [Y_pred]

# Labels = 'target','XGB SK API','XGB API'

# Plot.compare_regressions(dec,Labels,"GenPt/RecoPt",50,0.6,1.6, outfilename = 'XGB_compared')