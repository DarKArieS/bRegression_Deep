# #############################################################################
# Loss function
# #############################################################################


import numpy as np
import tensorflow as tf

'''
 ' Huber loss.
 ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
 ' https://en.wikipedia.org/wiki/Huber_loss
'''
def huber_loss(y_true, y_pred, clip_delta=0.5):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)

'''
 ' Same as above but returns the mean loss.
'''
def huber_loss_mean(y_true, y_pred, clip_delta=0.5):
  return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))
  

# #############################################################################
# Optimizer
# #############################################################################

from keras.optimizers import SGD
# sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
# default: keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False) ##really powerful, why?
sgd = SGD(lr=0.0001) # eth setting

# #############################################################################
# Model
# #############################################################################

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, LeakyReLU, ReLU
from keras.layers import Input, Add
from keras import regularizers

def Res_model_v0(nD=18):
	input 	= Input(shape=(nD,))
	layer1	= Dense(1000, activation='relu')(input)
	layer2	= Dense(1000, activation='relu')(layer1)
	add		= Add()([layer1, layer2])
	layer3	= Dense(1000, activation='relu')(add)
	output	= Dense(1)(layer3)
	
	model = Model(inputs=[input], outputs=output)
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse']) # 'acc' is no sense in regression
	return model
	
def Res_model_v1(nD=18):
	input 	= Input(shape=(nD,))
	layer1	= Dense(1000, activation='relu')(input)
	
	# dropout1= Dropout(0.5)(layer1)
	layer2_1= Dense(500, activation='relu')(layer1)
	layer2_2= Dense(500, activation='relu')(layer2_1)
	layer2_3= Dense(500, activation='relu')(layer2_2)
	add1	= Add()([layer2_1, layer2_3])
	
	layer3_1= Dense(500, activation='relu')(add1)
	layer3_2= Dense(500, activation='relu')(layer3_1)
	layer3_3= Dense(500, activation='relu')(layer3_2)
	add2	= Add()([layer3_1, layer3_3])
	
	layer4_1= Dense(500, activation='relu')(add2)
	layer4_2= Dense(500, activation='relu')(layer4_1)
	layer4_3= Dense(500, activation='relu')(layer4_2)
	add3	= Add()([layer4_1, layer4_3])
	
	layer5_1= Dense(500, activation='relu')(add3)
	layer5_2= Dense(500, activation='relu')(layer5_1)
	layer5_3= Dense(500, activation='relu')(layer5_2)
	add4	= Add()([layer5_1, layer5_3])
	
	layer6	= Dense(500, activation='relu')(add4)
	layer7	= Dense(500, activation='relu')(layer6)
	layer8	= Dense(500, activation='relu')(layer7)
	
	output	= Dense(1)(layer8)
	
	model = Model(inputs=[input], outputs=output)
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse']) # 'acc' is no sense in regression
	return model

def Res_model_v2(nD=18):
	input 	= Input(shape=(nD,))
	layer1	= Dense(250, activation='relu')(input)
	
	# dropout1= Dropout(0.5)(layer1)
	layer2_1= Dense(250, activation='relu')(layer1)
	layer2_2= Dense(250, activation='relu')(layer2_1)
	layer2_3= Dense(250, activation='relu')(layer2_2)
	add1	= Add()([layer2_1, layer2_3])
	
	layer3_1= Dense(250, activation='relu')(add1)
	layer3_2= Dense(250, activation='relu')(layer3_1)
	layer3_3= Dense(250, activation='relu')(layer3_2)
	add2	= Add()([layer3_1, layer3_3])
	
	layer4_1= Dense(250, activation='relu')(add2)
	layer4_2= Dense(250, activation='relu')(layer4_1)
	layer4_3= Dense(250, activation='relu')(layer4_2)
	add3	= Add()([layer4_1, layer4_3])
	
	layer5_1= Dense(250, activation='relu')(add3)
	layer5_2= Dense(250, activation='relu')(layer5_1)
	layer5_3= Dense(250, activation='relu')(layer5_2)
	add4	= Add()([layer5_1, layer5_3])
	
	layer6	= Dense(250, activation='relu')(add4)
	layer7	= Dense(250, activation='relu')(layer6)
	layer8	= Dense(250, activation='relu')(layer7)
	
	output	= Dense(1)(layer8)
	
	model = Model(inputs=input, outputs=output)
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse']) # 'acc' is no sense in regression
	return model

def Res_model_v3(nD=18):
	#Res_model_v1 + dropout
	input 	= Input(shape=(nD,))
	x	= Dense(1000, activation='relu')(input)
	
	for i in range(4):
		x = BatchNormalization()(x)
		identity1 = Dense(500, activation='relu')(x)
		x = Dense(500, activation='relu')(identity1)
		x = BatchNormalization()(x)
		x = Dropout(0.3)(x)
		x = Dense(500, activation='relu')(x)
		add	= Add()([x, identity1])
		x = ReLU()(add)
	
	x	= Dense(500, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.3)(x)
	x	= Dense(500, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.3)(x)
	x	= Dense(250, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.3)(x)
	
	output	= Dense(1)(x)
	
	model = Model(inputs=[input], outputs=output)
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse']) # 'acc' is no sense in regression
	return model
	
def Res_model_v4(nD=18):	
	#Res_model_v1 + l2 regularization + Huber
	input 	= Input(shape=(nD,))
	x	= Dense(1000, kernel_regularizer=regularizers.l2(0.01))(input)
	x	= LeakyReLU()(x)
	x	= BatchNormalization()(x)
	x	= Dense(500, kernel_regularizer=regularizers.l2(0.01))(x)
	x	= LeakyReLU()(x)
	
	
	for i in range(4):
		identity1 = BatchNormalization()(x)
		x = Dense(500, activation='relu')(identity1)
		x = LeakyReLU()(x)
		x = Dense(500, kernel_regularizer=regularizers.l2(0.01))(x)
		x = LeakyReLU()(x)
		x = BatchNormalization()(x)
		x = Dense(500, kernel_regularizer=regularizers.l2(0.01))(x)
		add	= Add()([x, identity1])
		x = LeakyReLU()(add)
		x = BatchNormalization()(x)
	

	x	= Dense(250, kernel_regularizer=regularizers.l2(0.01))(x)
	x = BatchNormalization()(x)
	
	output	= Dense(1)(x)
	
	model = Model(inputs=[input], outputs=output)
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse']) # 'acc' is no sense in regression
	return model
	
def Res_model2_v0(nD=18):
	input 	= Input(shape=(nD,))
	layer1	= Dense(1000, activation='relu')(input)
	layer2	= Dense(1000)(layer1)
	add		= Add()([layer1, layer2])
	Relu	= ReLU()(add)
	layer3	= Dense(1000, activation='relu')(Relu)
	output	= Dense(1)(layer3)
	
	model = Model(inputs=[input], outputs=output)
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse']) # 'acc' is no sense in regression
	return model
	
def model_3layer_v0(nD=18):
	# create model
	model = Sequential()
	model.add(Dense(2000, input_dim=nD, kernel_initializer = 'glorot_uniform'))
	model.add(ReLU())
	model.add(Dense(1000))
	model.add(ReLU())
	model.add(Dense(1))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse']) # 'acc' is no sense in regression
	return model
	
def model_16layer_v0(nD=18):
	# create model
	model = Sequential()
	model.add(Dense(1000, input_dim=nD, kernel_initializer = 'glorot_uniform'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(1))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse']) # 'acc' is no sense in regression
	return model

def model_3layer_1Dropout_v0(nD=18):
	# create model
	model = Sequential()
	model.add(Dense(2000, input_dim=nD, kernel_initializer = 'glorot_uniform'))
	model.add(ReLU())
	model.add(Dropout(0.5))
	model.add(Dense(1000))
	model.add(ReLU())
	model.add(Dense(1))
	# Compile model
	model.compile(loss='logcosh', optimizer='adam', metrics=['mse']) # 'acc' is no sense in regression
	return model
	
def model_3layer_1Dropout_v1(nD=18):
	# create model
	model = Sequential()
	model.add(Dense(2000, input_dim=nD, kernel_initializer = 'glorot_uniform'))
	model.add(ReLU())
	model.add(Dropout(0.5))
	model.add(Dense(1000))
	model.add(ReLU())
	model.add(Dense(1))
	# Compile model
	model.compile(loss=huber_loss, optimizer='adam', metrics=['mse']) # 'acc' is no sense in regression
	return model
	
def model_3layer_1Dropout_v2(nD=18):
	# create model
	model = Sequential()
	model.add(Dense(2000, input_dim=nD, kernel_initializer = 'glorot_uniform'))
	model.add(ReLU())
	model.add(Dropout(0.5))
	model.add(Dense(1000))
	model.add(ReLU())
	model.add(Dense(1))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse']) # 'acc' is no sense in regression
	return model
	
def eth_like_model(nD=18):
	# create model
	model = Sequential()
	model.add(Dense(1024, input_dim=nD))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(512))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(256))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(128))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss=huber_loss, optimizer='adam', metrics=['mse']) # 'acc' is no sense in regression
	return model

def eth_like_model_v0(nD=18):
	# create model
	model = Sequential()
	model.add(Dense(1024, input_dim=nD))
	model.add(ReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(512))
	model.add(ReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(256))
	model.add(ReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(128))
	model.add(ReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss=huber_loss, optimizer='adam', metrics=['mse']) # 'acc' is no sense in regression
	return model
	
def eth_like_model_v1(nD=18):
	# create model
	model = Sequential()
	model.add(Dense(1024, input_dim=nD))
	model.add(ReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(512))
	model.add(ReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(256))
	model.add(ReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(128))
	model.add(ReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mse', optimizer='adam', metrics=['mse']) # 'acc' is no sense in regression
	return model	

def eth_like_Huge_model():
	# create model
	model = Sequential()
	model.add(Dense(1024, input_dim=18))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(1024))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(1024))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(1024))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(1024))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(512))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(256))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(128))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss=huber_loss, optimizer='adam', metrics=['mse']) # 'acc' is no sense in regression
	return model

# #############################################################################
# Load Models
# #############################################################################	

from keras.models import load_model
def prediction_loadKeras(modelfile, testdata):
	Loadmodel_ = load_model(modelfile, custom_objects={'huber_loss': huber_loss})
	KerasPrediction_ = Loadmodel_.predict(testdata,verbose=1)
	KerasPrediction_ = KerasPrediction_.T
	return KerasPrediction_[0]