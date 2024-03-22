from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from keras import backend as K


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    sens = true_positives / (possible_positives + K.epsilon())
    return sens

def studymodel(activationf='relu', arquitetura = [5,5],lrate=1e-4,drop=0.2,x_train=[]):
  model = Sequential()

  ## Add the first hidden layer
  model.add(Dense(arquitetura[0], input_dim=x_train.shape[1], activation=activationf))
  model.add(Dropout(drop))

  hidden_layers = len(arquitetura)-1

  ## Add the remaining hidden layers (if any)
  if hidden_layers > 1:
      for i in range(hidden_layers - 1):
          model.add(Dense(arquitetura[i+1], activation=activationf))
          model.add(Dropout(drop))

  ## Add the output layer
  model.add(Dense(3, activation='softmax'))

  # Compile model
  optimizer = Adam(learning_rate=lrate)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
                metrics=[tfa.metrics.F1Score(average='macro',num_classes=3),precision,tf.keras.metrics.AUC(multi_label = True),sensitivity,'accuracy'])

  return model

def L1model(activationf='relu', arquitetura = [5,5],lrate=1e-4,drop=0.2,x_train=[]):
  model = Sequential()

  ## Add the first hidden layer
  model.add(Dense(arquitetura[0], input_dim=x_train.shape[1], activation=activationf,kernel_regularizer=keras.regularizers.L1(l1=0.001)))
  model.add(Dropout(drop))

  hidden_layers = len(arquitetura)-1

  ## Add the remaining hidden layers (if any)
  if hidden_layers > 1:
      for i in range(hidden_layers - 1):
          model.add(Dense(arquitetura[i+1], activation=activationf,kernel_regularizer=keras.regularizers.L1(l1=0.001)))
          model.add(Dropout(drop))

  ## Add the output layer
  model.add(Dense(3, activation='softmax'))

  # Compile model
  optimizer = Adam(learning_rate=lrate)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
                metrics=[tfa.metrics.F1Score(average='macro',num_classes=3),precision,tf.keras.metrics.AUC(multi_label = True),'accuracy',sensitivity])

  return model