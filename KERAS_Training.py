
# parameters
data_size = None
Variable_Type = 'KERAS_18var_4Dummies_cutSeries_GenOverReco_'
Model_name = 'Res_model_v4'
Input_dim = '22'
Num_epoch = 150
Batch_size = 512
# Other_cmd = '_SimpleScaled'
Other_cmd = '_SklearnScaled_fulldata'

print('Training:'+Variable_Type+Model_name+'_ep'+str(Num_epoch)+'_batch'+str(Batch_size) + Other_cmd)

# #############################################################################
# Load data
# #############################################################################
import pandas
import numpy as np
import modules.PreProcessing as PreProcessing

pre_dataframe = PreProcessing.read_dataset(data_size)
# pre_dataframe = PreProcessing.read_dataset(1200000)
# pre_dataframe = PreProcessing.read_dataset(10000)

#18_4dum_series_cut
# dataframe = pre_dataframe[['Jet_genjetPt','Jet_pt','Jet_eta','Jet_mt','Jet_leadTrackPt','Jet_leptonPtRel_new','Jet_leptonPt','Jet_leptonDeltaR','Jet_neHEF','Jet_neEmEF','Jet_vtxPt','Jet_vtxMass','Jet_vtx3dL','Jet_vtxNtrk','Jet_vtx3deL_new','nGoodPVs','Jet_PFMET','Jet_METDPhi','Jet_JetDR']]
dataframe = pre_dataframe[['Jet_genjetPt','Jet_pt','Jet_eta','Jet_mt','Jet_leadTrackPt','Jet_leptonPtRel_new','Jet_leptonPt','Jet_leptonDeltaR','Jet_neHEF','Jet_neEmEF','Jet_vtxPt','Jet_vtxMass','Jet_vtx3dL','Jet_vtxNtrk','Jet_vtx3deL_new','nGoodPVs','Jet_PFMET','Jet_METDPhi','Jet_JetDR','Jet_type','HasVertex', 'HasLepton', 'HasChTrk']]
dataset = dataframe.values.copy()
data = dataset[:,1:23]
target = dataset[:,0]/dataset[:,1]
# #############################################################################
import modules.KerasModel as KerasModel
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
# early_stop = EarlyStopping(monitor = "loss", mode = "min", patience = 5)

# estimator = KerasModel.eth_like_model(22)
exec('estimator = KerasModel.'+Model_name+'('+Input_dim+')')

# scaledata = PreProcessing.simple_scale_dataset(data)
scaledata = PreProcessing.sklearn_simple_scale(data)
# target = np.log1p(target) ##reverse: np.expm1(Y)

split = int(scaledata.shape[0]*0.9)

#training dataset
training_dataset = scaledata[0:split,:]
training_target = target[0:split]

Fit = estimator.fit(training_dataset,training_target,epochs = Num_epoch, batch_size = Batch_size, verbose=2)#, callbacks = [early_stop])
print(Fit.history)
d_History = pandas.DataFrame(Fit.history)
d_History.to_csv(Variable_Type+Model_name+'_ep'+str(Num_epoch)+'_batch'+str(Batch_size) + Other_cmd + '.txt',sep=' ',index=False)


# estimator.model.save('KERAS_18var_4Dummies_cutSeries_GenOverReco_model_v0_ep10_batch200.h5')
estimator.save(Variable_Type+Model_name+'_ep'+str(Num_epoch)+'_batch'+str(Batch_size) + Other_cmd + '.h5')

score = estimator.evaluate(training_dataset,training_target, verbose=2)
print(score)

print('Train Total Loss:', score[0])
print('Train Accuracy(MSE):', score[1])

#validate dataset
validate_dataset = scaledata[split:,:]
validate_target = target[split:]

validate_score = estimator.evaluate(validate_dataset,validate_target, verbose=2)
print('validate Total Loss:', validate_score[0])
print('validate Accuracy(MSE):', validate_score[1])

# print('original target test:')
# target_ori = dataset[:,0]/dataset[:,1]
# training_target_ori = target_ori[0:split]
# validate_target_ori = target_ori[split:]
# predictions_ori = np.expm1(predictions)
# val_predictions_ori = np.expm1(val_predictions)
# print('Train Accuracy(MSE):', mean_squared_error(training_target_ori, predictions))
# print('Validation Accuracy(MSE):', mean_squared_error(validate_target_ori, val_predictions))
