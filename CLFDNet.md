import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import glob as gb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from zipfile import ZipFile
import cv2

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D, LSTM, Dense, Concatenate, Attention, Dropout
from tensorflow.keras.optimizers import Adam

%==========================================================

# Load the dataset
df_path = '/content/dataset.csv'
df = pd.read_csv(df_path)

#-----------------------------------------------------------

# Display basic information about the dataset
print(df.info())

# Display descriptive statistics
#print(df.describe())
#==========================================================

# Step 4: Display the first few rows of the modified sheet

print(df.head())
print("\n")

#===========================================================

# Define features and labels
feature_columns = ['p', 'q', 'r', 'mx', 'my', 'mz']
label_columns = ['m1', 'm2', 'm3', 'm4']

#==========================================================

# Define a function to classify faults for all motors
def classify_fault(row):
    max_fault_value = max(row['m1'], row['m2'], row['m3'], row['m4'])
    if max_fault_value >= 0.4:
        return 'Fault 8'
    elif max_fault_value >= 0.35:
        return 'Fault 7'
    elif max_fault_value >= 0.30:
        return 'Fault 6'
    elif max_fault_value >= 0.25:
        return 'Fault 5'
    elif max_fault_value >= 0.20:
        return 'Fault 4'
    elif max_fault_value >= 0.15:
        return 'Fault 3'
    elif max_fault_value >= 0.10:
        return 'Fault 2'
    elif max_fault_value >= 0.05:
        return 'Fault 1'
    else:
        return 'No Fault'

# Apply the function to classify the faults
df['fault_classification'] = df.apply(classify_fault, axis=1)


# =======================================================

# Use LabelEncoder to encode the labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['fault_classification'])
num_classes = len(label_encoder.classes_)

# =======================================================

# Standardize your features
scaler = StandardScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])

# =======================================================

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df[feature_columns], df['label_encoded'], test_size=0.2, random_state=42)

# =======================================================

# Reshape your input data for the CNN and LSTM branches
X_train_cnn = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

X_train_lstm = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lstm = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# =======================================================

# One-Hot Encode your labels (if using categorical crossentropy loss)
from keras.utils import to_categorical

y_train_onehot = to_categorical(y_train, num_classes=num_classes)
y_test_onehot = to_categorical(y_test, num_classes=num_classes)

# =======================================================

# Display shapes to ensure everything is correct
print("X_train_cnn shape:", X_train_cnn.shape)
print("X_train_lstm shape:", X_train_lstm.shape)
print("y_train_onehot shape:", y_train_onehot.shape)

# =======================================================

# Define input shape for both CNN and LSTM
input_shape = (X_train_cnn.shape[1], 1)

# CNN branch
cnn_input = Input(shape=input_shape)

cnn1 = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_input)
cnn1 = GlobalAveragePooling1D()(cnn1)

cnn2 = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_input)
cnn2 = GlobalAveragePooling1D()(cnn2)

cnn3 = Conv1D(filters=48, kernel_size=5, activation='relu')(cnn_input)

cnn3 = GlobalAveragePooling1D()(cnn3)

cnn4 = Conv1D(filters=48, kernel_size=5, activation='relu')(cnn_input)

cnn4 = GlobalAveragePooling1D()(cnn4)

cnn5 = Conv1D(filters=64, kernel_size=3, activation='relu')(cnn_input)

cnn5 = GlobalAveragePooling1D()(cnn5)

cnn6 = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_input)
cnn6 = GlobalAveragePooling1D()(cnn6)

# Concatenate the outputs of CNN layers
merged_cnn = Concatenate()([cnn1, cnn2, cnn3, cnn4, cnn5, cnn6])

# Dense layers for CNN
dense_cnn = Dense(32, activation='relu')(merged_cnn)
dense_cnn = Dropout(0.05)(dense_cnn)  # Adding dropout layer
dense_cnn = Dense(32, activation='relu')(dense_cnn)

# LSTM branch
lstm_input = Input(shape=input_shape)
lstm = LSTM(units=32, return_sequences=True)(lstm_input)
lstm = Attention()([dense_cnn, lstm])
lstm = GlobalAveragePooling1D()(lstm)

# Concatenate the outputs of CNN and LSTM
merged = Concatenate()([dense_cnn, lstm])

# Dense layers for classification
dense = Dense(64, activation='relu')(merged)

# Corrected output layer
output = Dense(num_classes, activation='softmax')(dense)

# Create the model
model = Model(inputs=[cnn_input, lstm_input], outputs=output)


# Display the model summary
model.summary()

# =======================================================

_________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_32 (InputLayer)       [(None, 6, 1)]               0         []                            
                                                                                                  
 conv1d_109 (Conv1D)         (None, 4, 32)                128       ['input_32[0][0]']            
                                                                                                  
 conv1d_110 (Conv1D)         (None, 4, 32)                128       ['input_32[0][0]']            
                                                                                                  
 conv1d_111 (Conv1D)         (None, 2, 48)                288       ['input_32[0][0]']            
                                                                                                  
 conv1d_112 (Conv1D)         (None, 2, 48)                288       ['input_32[0][0]']            
                                                                                                  
 conv1d_113 (Conv1D)         (None, 4, 64)                256       ['input_32[0][0]']            
                                                                                                  
 conv1d_114 (Conv1D)         (None, 4, 32)                128       ['input_32[0][0]']            
                                                                                                  
 global_average_pooling1d_1  (None, 32)                   0         ['conv1d_109[0][0]']          
 36 (GlobalAveragePooling1D                                                                       
 )                                                                                                
                                                                                                  
 global_average_pooling1d_1  (None, 32)                   0         ['conv1d_110[0][0]']          
 37 (GlobalAveragePooling1D                                                                       
 )                                                                                                
                                                                                                  
 global_average_pooling1d_1  (None, 48)                   0         ['conv1d_111[0][0]']          
 38 (GlobalAveragePooling1D                                                                       
 )                                                                                                
                                                                                                  
 global_average_pooling1d_1  (None, 48)                   0         ['conv1d_112[0][0]']          
 39 (GlobalAveragePooling1D                                                                       
 )                                                                                                
                                                                                                  
 global_average_pooling1d_1  (None, 64)                   0         ['conv1d_113[0][0]']          
 40 (GlobalAveragePooling1D                                                                       
 )                                                                                                
                                                                                                  
 global_average_pooling1d_1  (None, 32)                   0         ['conv1d_114[0][0]']          
 41 (GlobalAveragePooling1D                                                                       
 )                                                                                                
                                                                                                  
 concatenate_30 (Concatenat  (None, 256)                  0         ['global_average_pooling1d_136
 e)                                                                 [0][0]',                      
                                                                     'global_average_pooling1d_137
                                                                    [0][0]',                      
                                                                     'global_average_pooling1d_138
                                                                    [0][0]',                      
                                                                     'global_average_pooling1d_139
                                                                    [0][0]',                      
                                                                     'global_average_pooling1d_140
                                                                    [0][0]',                      
                                                                     'global_average_pooling1d_141
                                                                    [0][0]']                      
                                                                                                  
 dense_67 (Dense)            (None, 32)                   8224      ['concatenate_30[0][0]']      
                                                                                                  
 dropout_30 (Dropout)        (None, 32)                   0         ['dense_67[0][0]']            
                                                                                                  
 input_33 (InputLayer)       [(None, 6, 1)]               0         []                            
                                                                                                  
 dense_68 (Dense)            (None, 32)                   1056      ['dropout_30[0][0]']          
                                                                                                  
 lstm_27 (LSTM)              (None, 6, 32)                4352      ['input_33[0][0]']            
                                                                                                  
 attention_3 (Attention)     (None, None, 32)             0         ['dense_68[0][0]',            
                                                                     'lstm_27[0][0]']             
                                                                                                  
 global_average_pooling1d_1  (None, 32)                   0         ['attention_3[0][0]']         
 42 (GlobalAveragePooling1D                                                                       
 )                                                                                                
                                                                                                  
 concatenate_31 (Concatenat  (None, 64)                   0         ['dense_68[0][0]',            
 e)                                                                  'global_average_pooling1d_142
                                                                    [0][0]']                      
                                                                                                  
 dense_69 (Dense)            (None, 64)                   4160      ['concatenate_31[0][0]']      
                                                                                                  
 dense_70 (Dense)            (None, 9)                    585       ['dense_69[0][0]']            
                                                                                                  
==================================================================================================
Total params: 19593 (76.54 KB)
Trainable params: 19593 (76.54 KB)
Non-trainable params: 0 (0.00 Byte)

#=================================================================================================

from tensorflow.keras.callbacks import ModelCheckpoint

# Define a custom learning rate
custom_learning_rate = 0.004

# Create an instance of the Adam optimizer with the custom learning rate
custom_optimizer = Adam(learning_rate=custom_learning_rate)

# Compile the model with the custom optimizer and appropriate loss function
model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define a ModelCheckpoint callback to save the best models based on validation accuracy
checkpoint = ModelCheckpoint(filepath='best_model_{epoch:02d}.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Train the model
history = model.fit([X_train_cnn, X_train_lstm], y_train, epochs=100, batch_size=32, validation_data=([X_test_cnn, X_test_lstm], y_test), callbacks=[checkpoint])

#=================================================================================================

Epoch 1/100
161/161 [==============================] - ETA: 0s - loss: 0.7602 - accuracy: 0.7319
Epoch 1: val_accuracy improved from -inf to 0.79144, saving model to best_model_01.h5
161/161 [==============================] - 7s 14ms/step - loss: 0.7602 - accuracy: 0.7319 - val_loss: 0.4250 - val_accuracy: 0.7914
Epoch 2/100
 14/161 [=>............................] - ETA: 1s - loss: 0.4039 - accuracy: 0.8170/usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(
160/161 [============================>.] - ETA: 0s - loss: 0.3971 - accuracy: 0.8467
Epoch 2: val_accuracy improved from 0.79144 to 0.84125, saving model to best_model_02.h5
161/161 [==============================] - 2s 10ms/step - loss: 0.3966 - accuracy: 0.8471 - val_loss: 0.3629 - val_accuracy: 0.8412
Epoch 3/100
159/161 [============================>.] - ETA: 0s - loss: 0.3846 - accuracy: 0.8473
Epoch 3: val_accuracy improved from 0.84125 to 0.87938, saving model to best_model_03.h5
161/161 [==============================] - 2s 12ms/step - loss: 0.3850 - accuracy: 0.8467 - val_loss: 0.3801 - val_accuracy: 0.8794
Epoch 4/100
157/161 [============================>.] - ETA: 0s - loss: 0.3242 - accuracy: 0.8929
Epoch 4: val_accuracy improved from 0.87938 to 0.88405, saving model to best_model_04.h5
161/161 [==============================] - 1s 9ms/step - loss: 0.3241 - accuracy: 0.8932 - val_loss: 0.3195 - val_accuracy: 0.8840
Epoch 5/100
160/161 [============================>.] - ETA: 0s - loss: 0.3163 - accuracy: 0.8977
Epoch 5: val_accuracy improved from 0.88405 to 0.88560, saving model to best_model_05.h5
161/161 [==============================] - 1s 9ms/step - loss: 0.3159 - accuracy: 0.8976 - val_loss: 0.3064 - val_accuracy: 0.8856
Epoch 6/100
155/161 [===========================>..] - ETA: 0s - loss: 0.2952 - accuracy: 0.9036
Epoch 6: val_accuracy improved from 0.88560 to 0.89105, saving model to best_model_06.h5
161/161 [==============================] - 1s 9ms/step - loss: 0.2933 - accuracy: 0.9043 - val_loss: 0.3367 - val_accuracy: 0.8911
Epoch 7/100
161/161 [==============================] - ETA: 0s - loss: 0.2877 - accuracy: 0.9052
Epoch 7: val_accuracy improved from 0.89105 to 0.92685, saving model to best_model_07.h5
161/161 [==============================] - 1s 9ms/step - loss: 0.2877 - accuracy: 0.9052 - val_loss: 0.2593 - val_accuracy: 0.9268
Epoch 8/100
158/161 [============================>.] - ETA: 0s - loss: 0.2691 - accuracy: 0.9084
Epoch 8: val_accuracy did not improve from 0.92685
161/161 [==============================] - 1s 8ms/step - loss: 0.2699 - accuracy: 0.9076 - val_loss: 0.3130 - val_accuracy: 0.8934
Epoch 9/100
159/161 [============================>.] - ETA: 0s - loss: 0.2732 - accuracy: 0.9068
Epoch 9: val_accuracy did not improve from 0.92685
161/161 [==============================] - 1s 9ms/step - loss: 0.2718 - accuracy: 0.9072 - val_loss: 0.2361 - val_accuracy: 0.9191
Epoch 10/100
157/161 [============================>.] - ETA: 0s - loss: 0.2570 - accuracy: 0.9134
Epoch 10: val_accuracy did not improve from 0.92685
161/161 [==============================] - 2s 10ms/step - loss: 0.2541 - accuracy: 0.9146 - val_loss: 0.2582 - val_accuracy: 0.9183
Epoch 11/100
155/161 [===========================>..] - ETA: 0s - loss: 0.2761 - accuracy: 0.9109
Epoch 11: val_accuracy did not improve from 0.92685
161/161 [==============================] - 2s 12ms/step - loss: 0.2725 - accuracy: 0.9120 - val_loss: 0.2277 - val_accuracy: 0.9198
Epoch 12/100
155/161 [===========================>..] - ETA: 0s - loss: 0.2172 - accuracy: 0.9288
Epoch 12: val_accuracy did not improve from 0.92685
161/161 [==============================] - 1s 8ms/step - loss: 0.2176 - accuracy: 0.9282 - val_loss: 0.2472 - val_accuracy: 0.9175
Epoch 13/100
159/161 [============================>.] - ETA: 0s - loss: 0.2237 - accuracy: 0.9292
Epoch 13: val_accuracy did not improve from 0.92685
161/161 [==============================] - 1s 9ms/step - loss: 0.2243 - accuracy: 0.9290 - val_loss: 0.2230 - val_accuracy: 0.9183
Epoch 14/100
161/161 [==============================] - ETA: 0s - loss: 0.2103 - accuracy: 0.9307
Epoch 14: val_accuracy did not improve from 0.92685
161/161 [==============================] - 1s 9ms/step - loss: 0.2103 - accuracy: 0.9307 - val_loss: 0.2188 - val_accuracy: 0.9222
Epoch 15/100
160/161 [============================>.] - ETA: 0s - loss: 0.2278 - accuracy: 0.9264
Epoch 15: val_accuracy did not improve from 0.92685
161/161 [==============================] - 2s 10ms/step - loss: 0.2275 - accuracy: 0.9266 - val_loss: 0.2127 - val_accuracy: 0.9237
Epoch 16/100
155/161 [===========================>..] - ETA: 0s - loss: 0.2016 - accuracy: 0.9317
Epoch 16: val_accuracy improved from 0.92685 to 0.93463, saving model to best_model_16.h5
161/161 [==============================] - 1s 9ms/step - loss: 0.2029 - accuracy: 0.9315 - val_loss: 0.2118 - val_accuracy: 0.9346
Epoch 17/100
155/161 [===========================>..] - ETA: 0s - loss: 0.2022 - accuracy: 0.9298
Epoch 17: val_accuracy did not improve from 0.93463
161/161 [==============================] - 1s 9ms/step - loss: 0.2037 - accuracy: 0.9296 - val_loss: 0.2146 - val_accuracy: 0.9268
Epoch 18/100
160/161 [============================>.] - ETA: 0s - loss: 0.1864 - accuracy: 0.9396
Epoch 18: val_accuracy improved from 0.93463 to 0.93852, saving model to best_model_18.h5
161/161 [==============================] - 2s 11ms/step - loss: 0.1868 - accuracy: 0.9393 - val_loss: 0.1762 - val_accuracy: 0.9385
Epoch 19/100
159/161 [============================>.] - ETA: 0s - loss: 0.1733 - accuracy: 0.9446
Epoch 19: val_accuracy did not improve from 0.93852
161/161 [==============================] - 2s 11ms/step - loss: 0.1746 - accuracy: 0.9445 - val_loss: 0.1891 - val_accuracy: 0.9339
Epoch 20/100
159/161 [============================>.] - ETA: 0s - loss: 0.1847 - accuracy: 0.9353
Epoch 20: val_accuracy did not improve from 0.93852
161/161 [==============================] - 1s 9ms/step - loss: 0.1836 - accuracy: 0.9358 - val_loss: 0.2022 - val_accuracy: 0.9237
Epoch 21/100
160/161 [============================>.] - ETA: 0s - loss: 0.1760 - accuracy: 0.9408
Epoch 21: val_accuracy improved from 0.93852 to 0.94630, saving model to best_model_21.h5
161/161 [==============================] - 1s 9ms/step - loss: 0.1765 - accuracy: 0.9406 - val_loss: 0.1747 - val_accuracy: 0.9463
Epoch 22/100
156/161 [============================>.] - ETA: 0s - loss: 0.1875 - accuracy: 0.9335
Epoch 22: val_accuracy did not improve from 0.94630
161/161 [==============================] - 1s 8ms/step - loss: 0.1872 - accuracy: 0.9327 - val_loss: 0.1846 - val_accuracy: 0.9385
Epoch 23/100
161/161 [==============================] - ETA: 0s - loss: 0.1772 - accuracy: 0.9364
Epoch 23: val_accuracy did not improve from 0.94630
161/161 [==============================] - 1s 8ms/step - loss: 0.1772 - accuracy: 0.9364 - val_loss: 0.1789 - val_accuracy: 0.9323
Epoch 24/100
155/161 [===========================>..] - ETA: 0s - loss: 0.1619 - accuracy: 0.9440
Epoch 24: val_accuracy did not improve from 0.94630
161/161 [==============================] - 1s 8ms/step - loss: 0.1627 - accuracy: 0.9438 - val_loss: 0.1938 - val_accuracy: 0.9268
Epoch 25/100
161/161 [==============================] - ETA: 0s - loss: 0.1581 - accuracy: 0.9453
Epoch 25: val_accuracy did not improve from 0.94630
161/161 [==============================] - 1s 9ms/step - loss: 0.1581 - accuracy: 0.9453 - val_loss: 0.2207 - val_accuracy: 0.9097
Epoch 26/100
157/161 [============================>.] - ETA: 0s - loss: 0.2280 - accuracy: 0.9226
Epoch 26: val_accuracy did not improve from 0.94630
161/161 [==============================] - 2s 11ms/step - loss: 0.2259 - accuracy: 0.9235 - val_loss: 0.1958 - val_accuracy: 0.9370
Epoch 27/100
155/161 [===========================>..] - ETA: 0s - loss: 0.1778 - accuracy: 0.9369
Epoch 27: val_accuracy did not improve from 0.94630
161/161 [==============================] - 2s 11ms/step - loss: 0.1784 - accuracy: 0.9364 - val_loss: 0.1914 - val_accuracy: 0.9307
Epoch 28/100
158/161 [============================>.] - ETA: 0s - loss: 0.1554 - accuracy: 0.9476
Epoch 28: val_accuracy improved from 0.94630 to 0.94708, saving model to best_model_28.h5
161/161 [==============================] - 1s 9ms/step - loss: 0.1571 - accuracy: 0.9469 - val_loss: 0.1429 - val_accuracy: 0.9471
Epoch 29/100
160/161 [============================>.] - ETA: 0s - loss: 0.1642 - accuracy: 0.9414
Epoch 29: val_accuracy did not improve from 0.94708
161/161 [==============================] - 1s 9ms/step - loss: 0.1638 - accuracy: 0.9416 - val_loss: 0.1515 - val_accuracy: 0.9416
Epoch 30/100
157/161 [============================>.] - ETA: 0s - loss: 0.1506 - accuracy: 0.9453
Epoch 30: val_accuracy did not improve from 0.94708
161/161 [==============================] - 2s 12ms/step - loss: 0.1517 - accuracy: 0.9447 - val_loss: 0.1748 - val_accuracy: 0.9354
Epoch 31/100
161/161 [==============================] - ETA: 0s - loss: 0.1494 - accuracy: 0.9484
Epoch 31: val_accuracy did not improve from 0.94708
161/161 [==============================] - 2s 14ms/step - loss: 0.1494 - accuracy: 0.9484 - val_loss: 0.1489 - val_accuracy: 0.9401
Epoch 32/100
159/161 [============================>.] - ETA: 0s - loss: 0.1591 - accuracy: 0.9428
Epoch 32: val_accuracy did not improve from 0.94708
161/161 [==============================] - 1s 9ms/step - loss: 0.1584 - accuracy: 0.9430 - val_loss: 0.2515 - val_accuracy: 0.9043
Epoch 33/100
159/161 [============================>.] - ETA: 0s - loss: 0.1452 - accuracy: 0.9497
Epoch 33: val_accuracy did not improve from 0.94708
161/161 [==============================] - 2s 11ms/step - loss: 0.1457 - accuracy: 0.9494 - val_loss: 0.1322 - val_accuracy: 0.9455
Epoch 34/100
160/161 [============================>.] - ETA: 0s - loss: 0.1627 - accuracy: 0.9432
Epoch 34: val_accuracy did not improve from 0.94708
161/161 [==============================] - 2s 11ms/step - loss: 0.1628 - accuracy: 0.9430 - val_loss: 0.1976 - val_accuracy: 0.9214
Epoch 35/100
155/161 [===========================>..] - ETA: 0s - loss: 0.1575 - accuracy: 0.9407
Epoch 35: val_accuracy did not improve from 0.94708
161/161 [==============================] - 1s 9ms/step - loss: 0.1589 - accuracy: 0.9399 - val_loss: 0.1584 - val_accuracy: 0.9385
Epoch 36/100
155/161 [===========================>..] - ETA: 0s - loss: 0.1459 - accuracy: 0.9488
Epoch 36: val_accuracy did not improve from 0.94708
161/161 [==============================] - 1s 9ms/step - loss: 0.1459 - accuracy: 0.9484 - val_loss: 0.1919 - val_accuracy: 0.9307
Epoch 37/100
155/161 [===========================>..] - ETA: 0s - loss: 0.1429 - accuracy: 0.9492
Epoch 37: val_accuracy did not improve from 0.94708
161/161 [==============================] - 1s 8ms/step - loss: 0.1418 - accuracy: 0.9500 - val_loss: 0.1477 - val_accuracy: 0.9447
Epoch 38/100
161/161 [==============================] - ETA: 0s - loss: 0.1328 - accuracy: 0.9533
Epoch 38: val_accuracy did not improve from 0.94708
161/161 [==============================] - 1s 9ms/step - loss: 0.1328 - accuracy: 0.9533 - val_loss: 0.1510 - val_accuracy: 0.9409
Epoch 39/100
159/161 [============================>.] - ETA: 0s - loss: 0.1667 - accuracy: 0.9442
Epoch 39: val_accuracy improved from 0.94708 to 0.94786, saving model to best_model_39.h5
161/161 [==============================] - 1s 9ms/step - loss: 0.1655 - accuracy: 0.9445 - val_loss: 0.1401 - val_accuracy: 0.9479
Epoch 40/100
160/161 [============================>.] - ETA: 0s - loss: 0.1347 - accuracy: 0.9518
Epoch 40: val_accuracy did not improve from 0.94786
161/161 [==============================] - 1s 9ms/step - loss: 0.1351 - accuracy: 0.9515 - val_loss: 0.1544 - val_accuracy: 0.9409
Epoch 41/100
161/161 [==============================] - ETA: 0s - loss: 0.1336 - accuracy: 0.9519
Epoch 41: val_accuracy improved from 0.94786 to 0.95019, saving model to best_model_41.h5
161/161 [==============================] - 2s 12ms/step - loss: 0.1336 - accuracy: 0.9519 - val_loss: 0.1414 - val_accuracy: 0.9502
Epoch 42/100
158/161 [============================>.] - ETA: 0s - loss: 0.1216 - accuracy: 0.9559
Epoch 42: val_accuracy improved from 0.95019 to 0.95175, saving model to best_model_42.h5
161/161 [==============================] - 2s 11ms/step - loss: 0.1216 - accuracy: 0.9556 - val_loss: 0.1261 - val_accuracy: 0.9518
Epoch 43/100
158/161 [============================>.] - ETA: 0s - loss: 0.1169 - accuracy: 0.9579
Epoch 43: val_accuracy did not improve from 0.95175
161/161 [==============================] - 1s 9ms/step - loss: 0.1171 - accuracy: 0.9580 - val_loss: 0.1385 - val_accuracy: 0.9494
Epoch 44/100
160/161 [============================>.] - ETA: 0s - loss: 0.1341 - accuracy: 0.9514
Epoch 44: val_accuracy improved from 0.95175 to 0.95564, saving model to best_model_44.h5
161/161 [==============================] - 1s 9ms/step - loss: 0.1338 - accuracy: 0.9515 - val_loss: 0.1210 - val_accuracy: 0.9556
Epoch 45/100
161/161 [==============================] - ETA: 0s - loss: 0.1549 - accuracy: 0.9459
Epoch 45: val_accuracy improved from 0.95564 to 0.95875, saving model to best_model_45.h5
161/161 [==============================] - 2s 9ms/step - loss: 0.1549 - accuracy: 0.9459 - val_loss: 0.1296 - val_accuracy: 0.9588
Epoch 46/100
160/161 [============================>.] - ETA: 0s - loss: 0.1425 - accuracy: 0.9529
Epoch 46: val_accuracy did not improve from 0.95875
161/161 [==============================] - 1s 9ms/step - loss: 0.1421 - accuracy: 0.9531 - val_loss: 0.1264 - val_accuracy: 0.9533
Epoch 47/100
157/161 [============================>.] - ETA: 0s - loss: 0.1284 - accuracy: 0.9550
Epoch 47: val_accuracy did not improve from 0.95875
161/161 [==============================] - 1s 9ms/step - loss: 0.1271 - accuracy: 0.9558 - val_loss: 0.1085 - val_accuracy: 0.9572
Epoch 48/100
157/161 [============================>.] - ETA: 0s - loss: 0.1047 - accuracy: 0.9618
Epoch 48: val_accuracy did not improve from 0.95875
161/161 [==============================] - 1s 9ms/step - loss: 0.1045 - accuracy: 0.9619 - val_loss: 0.1427 - val_accuracy: 0.9463
Epoch 49/100
159/161 [============================>.] - ETA: 0s - loss: 0.1229 - accuracy: 0.9579
Epoch 49: val_accuracy did not improve from 0.95875
161/161 [==============================] - 2s 11ms/step - loss: 0.1232 - accuracy: 0.9578 - val_loss: 0.1737 - val_accuracy: 0.9409
Epoch 50/100
160/161 [============================>.] - ETA: 0s - loss: 0.1445 - accuracy: 0.9500
Epoch 50: val_accuracy did not improve from 0.95875
161/161 [==============================] - 2s 11ms/step - loss: 0.1441 - accuracy: 0.9502 - val_loss: 0.1552 - val_accuracy: 0.9276
Epoch 51/100
158/161 [============================>.] - ETA: 0s - loss: 0.1193 - accuracy: 0.9569
Epoch 51: val_accuracy did not improve from 0.95875
161/161 [==============================] - 1s 9ms/step - loss: 0.1187 - accuracy: 0.9572 - val_loss: 0.1316 - val_accuracy: 0.9541
Epoch 52/100
161/161 [==============================] - ETA: 0s - loss: 0.1212 - accuracy: 0.9580
Epoch 52: val_accuracy did not improve from 0.95875
161/161 [==============================] - 1s 9ms/step - loss: 0.1212 - accuracy: 0.9580 - val_loss: 0.1484 - val_accuracy: 0.9486
Epoch 53/100
160/161 [============================>.] - ETA: 0s - loss: 0.1079 - accuracy: 0.9623
Epoch 53: val_accuracy did not improve from 0.95875
161/161 [==============================] - 1s 9ms/step - loss: 0.1077 - accuracy: 0.9624 - val_loss: 0.1275 - val_accuracy: 0.9564
Epoch 54/100
159/161 [============================>.] - ETA: 0s - loss: 0.1220 - accuracy: 0.9574
Epoch 54: val_accuracy improved from 0.95875 to 0.96265, saving model to best_model_54.h5
161/161 [==============================] - 1s 9ms/step - loss: 0.1214 - accuracy: 0.9576 - val_loss: 0.1147 - val_accuracy: 0.9626
Epoch 55/100
158/161 [============================>.] - ETA: 0s - loss: 0.1092 - accuracy: 0.9604
Epoch 55: val_accuracy did not improve from 0.96265
161/161 [==============================] - 1s 9ms/step - loss: 0.1087 - accuracy: 0.9609 - val_loss: 0.1298 - val_accuracy: 0.9549
Epoch 56/100
160/161 [============================>.] - ETA: 0s - loss: 0.1057 - accuracy: 0.9611
Epoch 56: val_accuracy improved from 0.96265 to 0.96420, saving model to best_model_56.h5
161/161 [==============================] - 1s 9ms/step - loss: 0.1059 - accuracy: 0.9611 - val_loss: 0.1003 - val_accuracy: 0.9642
Epoch 57/100
159/161 [============================>.] - ETA: 0s - loss: 0.1195 - accuracy: 0.9556
Epoch 57: val_accuracy did not improve from 0.96420
161/161 [==============================] - 2s 13ms/step - loss: 0.1188 - accuracy: 0.9558 - val_loss: 0.1453 - val_accuracy: 0.9580
Epoch 58/100
160/161 [============================>.] - ETA: 0s - loss: 0.1341 - accuracy: 0.9553
Epoch 58: val_accuracy did not improve from 0.96420
161/161 [==============================] - 2s 10ms/step - loss: 0.1338 - accuracy: 0.9554 - val_loss: 0.1751 - val_accuracy: 0.9362
Epoch 59/100
160/161 [============================>.] - ETA: 0s - loss: 0.1025 - accuracy: 0.9637
Epoch 59: val_accuracy did not improve from 0.96420
161/161 [==============================] - 1s 9ms/step - loss: 0.1023 - accuracy: 0.9636 - val_loss: 0.1451 - val_accuracy: 0.9416
Epoch 60/100
161/161 [==============================] - ETA: 0s - loss: 0.1093 - accuracy: 0.9589
Epoch 60: val_accuracy did not improve from 0.96420
161/161 [==============================] - 2s 10ms/step - loss: 0.1093 - accuracy: 0.9589 - val_loss: 0.1017 - val_accuracy: 0.9626
Epoch 61/100
155/161 [===========================>..] - ETA: 0s - loss: 0.1348 - accuracy: 0.9512
Epoch 61: val_accuracy did not improve from 0.96420
161/161 [==============================] - 2s 10ms/step - loss: 0.1338 - accuracy: 0.9515 - val_loss: 0.2192 - val_accuracy: 0.9315
Epoch 62/100
159/161 [============================>.] - ETA: 0s - loss: 0.1136 - accuracy: 0.9603
Epoch 62: val_accuracy did not improve from 0.96420
161/161 [==============================] - 1s 9ms/step - loss: 0.1132 - accuracy: 0.9605 - val_loss: 0.1538 - val_accuracy: 0.9393
Epoch 63/100
159/161 [============================>.] - ETA: 0s - loss: 0.1589 - accuracy: 0.9465
Epoch 63: val_accuracy did not improve from 0.96420
161/161 [==============================] - 1s 9ms/step - loss: 0.1586 - accuracy: 0.9463 - val_loss: 0.1117 - val_accuracy: 0.9588
Epoch 64/100
156/161 [============================>.] - ETA: 0s - loss: 0.0941 - accuracy: 0.9679
Epoch 64: val_accuracy did not improve from 0.96420
161/161 [==============================] - 2s 10ms/step - loss: 0.0943 - accuracy: 0.9675 - val_loss: 0.1249 - val_accuracy: 0.9440
Epoch 65/100
159/161 [============================>.] - ETA: 0s - loss: 0.0890 - accuracy: 0.9689
Epoch 65: val_accuracy did not improve from 0.96420
161/161 [==============================] - 2s 12ms/step - loss: 0.0887 - accuracy: 0.9687 - val_loss: 0.1255 - val_accuracy: 0.9440
Epoch 66/100
159/161 [============================>.] - ETA: 0s - loss: 0.1020 - accuracy: 0.9629
Epoch 66: val_accuracy did not improve from 0.96420
161/161 [==============================] - 1s 9ms/step - loss: 0.1012 - accuracy: 0.9632 - val_loss: 0.1298 - val_accuracy: 0.9494
Epoch 67/100
159/161 [============================>.] - ETA: 0s - loss: 0.1099 - accuracy: 0.9603
Epoch 67: val_accuracy did not improve from 0.96420
161/161 [==============================] - 1s 9ms/step - loss: 0.1093 - accuracy: 0.9605 - val_loss: 0.1056 - val_accuracy: 0.9642
Epoch 68/100
155/161 [===========================>..] - ETA: 0s - loss: 0.0946 - accuracy: 0.9681
Epoch 68: val_accuracy did not improve from 0.96420
161/161 [==============================] - 1s 9ms/step - loss: 0.0953 - accuracy: 0.9675 - val_loss: 0.2061 - val_accuracy: 0.9206
Epoch 69/100
156/161 [============================>.] - ETA: 0s - loss: 0.1207 - accuracy: 0.9567
Epoch 69: val_accuracy did not improve from 0.96420
161/161 [==============================] - 1s 9ms/step - loss: 0.1188 - accuracy: 0.9574 - val_loss: 0.1039 - val_accuracy: 0.9603
Epoch 70/100
157/161 [============================>.] - ETA: 0s - loss: 0.1164 - accuracy: 0.9580
Epoch 70: val_accuracy improved from 0.96420 to 0.96887, saving model to best_model_70.h5
161/161 [==============================] - 1s 9ms/step - loss: 0.1169 - accuracy: 0.9576 - val_loss: 0.0917 - val_accuracy: 0.9689
Epoch 71/100
157/161 [============================>.] - ETA: 0s - loss: 0.0841 - accuracy: 0.9691
Epoch 71: val_accuracy did not improve from 0.96887
161/161 [==============================] - 1s 9ms/step - loss: 0.0833 - accuracy: 0.9693 - val_loss: 0.0992 - val_accuracy: 0.9611
Epoch 72/100
160/161 [============================>.] - ETA: 0s - loss: 0.0810 - accuracy: 0.9688
Epoch 72: val_accuracy did not improve from 0.96887
161/161 [==============================] - 2s 11ms/step - loss: 0.0808 - accuracy: 0.9689 - val_loss: 0.1125 - val_accuracy: 0.9572
Epoch 73/100
160/161 [============================>.] - ETA: 0s - loss: 0.0845 - accuracy: 0.9699
Epoch 73: val_accuracy did not improve from 0.96887
161/161 [==============================] - 2s 12ms/step - loss: 0.0843 - accuracy: 0.9700 - val_loss: 0.0790 - val_accuracy: 0.9689
Epoch 74/100
159/161 [============================>.] - ETA: 0s - loss: 0.0939 - accuracy: 0.9656
Epoch 74: val_accuracy did not improve from 0.96887
161/161 [==============================] - 1s 9ms/step - loss: 0.0937 - accuracy: 0.9658 - val_loss: 0.1602 - val_accuracy: 0.9409
Epoch 75/100
157/161 [============================>.] - ETA: 0s - loss: 0.0954 - accuracy: 0.9660
Epoch 75: val_accuracy did not improve from 0.96887
161/161 [==============================] - 1s 9ms/step - loss: 0.0958 - accuracy: 0.9658 - val_loss: 0.1205 - val_accuracy: 0.9595
Epoch 76/100
160/161 [============================>.] - ETA: 0s - loss: 0.0795 - accuracy: 0.9711
Epoch 76: val_accuracy did not improve from 0.96887
161/161 [==============================] - 1s 9ms/step - loss: 0.0793 - accuracy: 0.9712 - val_loss: 0.1075 - val_accuracy: 0.9580
Epoch 77/100
158/161 [============================>.] - ETA: 0s - loss: 0.1524 - accuracy: 0.9553
Epoch 77: val_accuracy did not improve from 0.96887
161/161 [==============================] - 1s 9ms/step - loss: 0.1524 - accuracy: 0.9552 - val_loss: 0.1085 - val_accuracy: 0.9681
Epoch 78/100
159/161 [============================>.] - ETA: 0s - loss: 0.0880 - accuracy: 0.9672
Epoch 78: val_accuracy did not improve from 0.96887
161/161 [==============================] - 1s 9ms/step - loss: 0.0882 - accuracy: 0.9671 - val_loss: 0.1214 - val_accuracy: 0.9626
Epoch 79/100
156/161 [============================>.] - ETA: 0s - loss: 0.0812 - accuracy: 0.9694
Epoch 79: val_accuracy did not improve from 0.96887
161/161 [==============================] - 1s 9ms/step - loss: 0.0803 - accuracy: 0.9696 - val_loss: 0.1716 - val_accuracy: 0.9362
Epoch 80/100
159/161 [============================>.] - ETA: 0s - loss: 0.1047 - accuracy: 0.9642
Epoch 80: val_accuracy did not improve from 0.96887
161/161 [==============================] - 2s 12ms/step - loss: 0.1041 - accuracy: 0.9644 - val_loss: 0.2083 - val_accuracy: 0.9175
Epoch 81/100
156/161 [============================>.] - ETA: 0s - loss: 0.0789 - accuracy: 0.9706
Epoch 81: val_accuracy did not improve from 0.96887
161/161 [==============================] - 2s 10ms/step - loss: 0.0783 - accuracy: 0.9706 - val_loss: 0.0870 - val_accuracy: 0.9681
Epoch 82/100
157/161 [============================>.] - ETA: 0s - loss: 0.1002 - accuracy: 0.9644
Epoch 82: val_accuracy did not improve from 0.96887
161/161 [==============================] - 1s 9ms/step - loss: 0.1001 - accuracy: 0.9646 - val_loss: 0.1267 - val_accuracy: 0.9580
Epoch 83/100
158/161 [============================>.] - ETA: 0s - loss: 0.1131 - accuracy: 0.9634
Epoch 83: val_accuracy did not improve from 0.96887
161/161 [==============================] - 1s 9ms/step - loss: 0.1151 - accuracy: 0.9628 - val_loss: 0.2005 - val_accuracy: 0.9300
Epoch 84/100
160/161 [============================>.] - ETA: 0s - loss: 0.1042 - accuracy: 0.9637
Epoch 84: val_accuracy did not improve from 0.96887
161/161 [==============================] - 1s 9ms/step - loss: 0.1042 - accuracy: 0.9636 - val_loss: 0.1154 - val_accuracy: 0.9588
Epoch 85/100
160/161 [============================>.] - ETA: 0s - loss: 0.0700 - accuracy: 0.9756
Epoch 85: val_accuracy did not improve from 0.96887
161/161 [==============================] - 1s 9ms/step - loss: 0.0699 - accuracy: 0.9757 - val_loss: 0.1147 - val_accuracy: 0.9564
Epoch 86/100
155/161 [===========================>..] - ETA: 0s - loss: 0.0683 - accuracy: 0.9764
Epoch 86: val_accuracy did not improve from 0.96887
161/161 [==============================] - 1s 9ms/step - loss: 0.0703 - accuracy: 0.9761 - val_loss: 0.0951 - val_accuracy: 0.9611
Epoch 87/100
159/161 [============================>.] - ETA: 0s - loss: 0.0661 - accuracy: 0.9760
Epoch 87: val_accuracy improved from 0.96887 to 0.97121, saving model to best_model_87.h5
161/161 [==============================] - 1s 9ms/step - loss: 0.0657 - accuracy: 0.9761 - val_loss: 0.0931 - val_accuracy: 0.9712
Epoch 88/100
158/161 [============================>.] - ETA: 0s - loss: 0.0740 - accuracy: 0.9723
Epoch 88: val_accuracy did not improve from 0.97121
161/161 [==============================] - 2s 13ms/step - loss: 0.0742 - accuracy: 0.9722 - val_loss: 0.1272 - val_accuracy: 0.9556
Epoch 89/100
156/161 [============================>.] - ETA: 0s - loss: 0.0812 - accuracy: 0.9690
Epoch 89: val_accuracy did not improve from 0.97121
161/161 [==============================] - 2s 10ms/step - loss: 0.0813 - accuracy: 0.9691 - val_loss: 0.0995 - val_accuracy: 0.9619
Epoch 90/100
161/161 [==============================] - ETA: 0s - loss: 0.0864 - accuracy: 0.9698
Epoch 90: val_accuracy did not improve from 0.97121
161/161 [==============================] - 1s 9ms/step - loss: 0.0864 - accuracy: 0.9698 - val_loss: 0.1337 - val_accuracy: 0.9440
Epoch 91/100
160/161 [============================>.] - ETA: 0s - loss: 0.0967 - accuracy: 0.9660
Epoch 91: val_accuracy improved from 0.97121 to 0.97198, saving model to best_model_91.h5
161/161 [==============================] - 1s 9ms/step - loss: 0.0969 - accuracy: 0.9658 - val_loss: 0.0941 - val_accuracy: 0.9720
Epoch 92/100
158/161 [============================>.] - ETA: 0s - loss: 0.0746 - accuracy: 0.9709
Epoch 92: val_accuracy did not improve from 0.97198
161/161 [==============================] - 1s 9ms/step - loss: 0.0757 - accuracy: 0.9704 - val_loss: 0.1499 - val_accuracy: 0.9479
Epoch 93/100
156/161 [============================>.] - ETA: 0s - loss: 0.0822 - accuracy: 0.9700
Epoch 93: val_accuracy did not improve from 0.97198
161/161 [==============================] - 2s 12ms/step - loss: 0.0833 - accuracy: 0.9700 - val_loss: 0.1446 - val_accuracy: 0.9580
Epoch 94/100
160/161 [============================>.] - ETA: 0s - loss: 0.0985 - accuracy: 0.9674
Epoch 94: val_accuracy did not improve from 0.97198
161/161 [==============================] - 1s 9ms/step - loss: 0.0982 - accuracy: 0.9675 - val_loss: 0.0969 - val_accuracy: 0.9658
Epoch 95/100
161/161 [==============================] - ETA: 0s - loss: 0.0886 - accuracy: 0.9677
Epoch 95: val_accuracy did not improve from 0.97198
161/161 [==============================] - 2s 10ms/step - loss: 0.0886 - accuracy: 0.9677 - val_loss: 0.0817 - val_accuracy: 0.9696
Epoch 96/100
159/161 [============================>.] - ETA: 0s - loss: 0.0605 - accuracy: 0.9778
Epoch 96: val_accuracy improved from 0.97198 to 0.97354, saving model to best_model_96.h5
161/161 [==============================] - 2s 13ms/step - loss: 0.0601 - accuracy: 0.9780 - val_loss: 0.0833 - val_accuracy: 0.9735
Epoch 97/100
161/161 [==============================] - ETA: 0s - loss: 0.0689 - accuracy: 0.9728
Epoch 97: val_accuracy did not improve from 0.97354
161/161 [==============================] - 1s 9ms/step - loss: 0.0689 - accuracy: 0.9728 - val_loss: 0.1175 - val_accuracy: 0.9611
Epoch 98/100
155/161 [===========================>..] - ETA: 0s - loss: 0.0973 - accuracy: 0.9657
Epoch 98: val_accuracy did not improve from 0.97354
161/161 [==============================] - 1s 9ms/step - loss: 0.0975 - accuracy: 0.9652 - val_loss: 0.1481 - val_accuracy: 0.9463
Epoch 99/100
161/161 [==============================] - ETA: 0s - loss: 0.0739 - accuracy: 0.9743
Epoch 99: val_accuracy did not improve from 0.97354
161/161 [==============================] - 1s 9ms/step - loss: 0.0739 - accuracy: 0.9743 - val_loss: 0.0991 - val_accuracy: 0.9626
Epoch 100/100
159/161 [============================>.] - ETA: 0s - loss: 0.0551 - accuracy: 0.9801
Epoch 100: val_accuracy did not improve from 0.97354
161/161 [==============================] - 1s 9ms/step - loss: 0.0555 - accuracy: 0.9800 - val_loss: 0.0954 - val_accuracy: 0.9681

#=======================================================================================================

# Evaluate the model
test_loss, test_accuracy = model.evaluate([X_test_cnn, X_test_lstm], y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
#=======================================================================================================
41/41 [==============================] - 0s 3ms/step - loss: 0.0954 - accuracy: 0.9681
Test Loss: 0.09544660896062851
Test Accuracy: 0.9680933952331543
#=======================================================================================================
# Predict on the test data
predictions = model.predict([X_test_cnn, X_test_lstm])

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Decode labels using LabelEncoder
decoded_predicted_labels = label_encoder.inverse_transform(predicted_labels)

# Compare predicted labels with actual labels
comparison_df = pd.DataFrame({'Actual': label_encoder.inverse_transform(y_test), 'Predicted': decoded_predicted_labels})
print(comparison_df)

#=======================================================================================================

41/41 [==============================] - 1s 3ms/step
        Actual Predicted
0     No Fault  No Fault
1     No Fault  No Fault
2      Fault 3   Fault 3
3     No Fault  No Fault
4     No Fault  No Fault
...        ...       ...
1280  No Fault  No Fault
1281  No Fault  No Fault
1282   Fault 3   Fault 3
1283  No Fault  No Fault
1284   Fault 1   Fault 1

[1285 rows x 2 columns]

#=======================================================================================================

# Define a function to map fault classifications to corresponding motors
def map_fault_to_motor(fault):
    if fault == 'Fault 8':
        return 'm3'  # Assuming Fault 8 corresponds to motor m3
    elif fault in ['Fault 1', 'Fault 2', 'Fault 3', 'Fault 4', 'Fault 5', 'Fault 6', 'Fault 7']:
        return 'm1'  # Assuming Faults 1 to 7 correspond to motor m1
    else:
        return None  # Handle the case where fault is not recognized or not applicable to any motor

# Map predicted fault classifications to corresponding motors
df['Motor'] = df['fault_classification'].apply(map_fault_to_motor)

# Print the DataFrame with fault classifications and corresponding motors
print(df[['fault_classification', 'Motor']])

#=======================================================================================================

   fault_classification Motor
0                No Fault  None
1                No Fault  None
2                No Fault  None
3                No Fault  None
4                No Fault  None
...                   ...   ...
6419              Fault 8    m3
6420              Fault 8    m3
6421              Fault 8    m3
6422              Fault 8    m3
6423              Fault 8    m3


#=======================================================================================================

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get predictions on the test data
test_predictions = model.predict([X_test_cnn, X_test_lstm])
test_predicted_labels = np.argmax(test_predictions, axis=1)

# Convert one-hot encoded true labels to categorical labels
decoded_actual_labels = label_encoder.inverse_transform(y_test)

# Convert predicted labels to categorical labels
decoded_predicted_labels = label_encoder.inverse_transform(test_predicted_labels)

# Define class names (fault types)
class_names = ['No Fault', 'Fault 1', 'Fault 2', 'Fault 3', 'Fault 4', 'Fault 5', 'Fault 6', 'Fault 7', 'Fault 8']

# Create confusion matrix
conf_matrix = confusion_matrix(decoded_actual_labels, decoded_predicted_labels, labels=class_names)

# Set seaborn style
sns.set(style='whitegrid', font_scale=1.2)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

plt.tight_layout()

# Save the figure with dpi 400
plt.savefig('confusion_matrix.png', dpi=400)

plt.show()

#=======================================================================================================
<img width="1176" height="976" alt="download" src="https://github.com/user-attachments/assets/3cb17247-1bbc-4664-88ce-ad7f8ae2a808" />

#=======================================================================================================
from sklearn.metrics import classification_report

# Predict on the test set
y_pred = model.predict([X_test_cnn, X_test_lstm])

# Convert predictions to class labels
y_pred_classes = y_pred.argmax(axis=1)

# Decode class labels using the inverse of LabelEncoder
decoded_predictions = label_encoder.inverse_transform(y_pred_classes)

# Convert one-hot encoded fault values to actual values
decoded_actual_values = label_encoder.inverse_transform(y_test)

# Generate classification report
report = classification_report(decoded_actual_values, decoded_predictions)
print("Classification Report:\n", report)

#=======================================================================================================

41/41 [==============================] - 0s 3ms/step
Classification Report:
               precision    recall  f1-score   support

     Fault 1       0.99      1.00      0.99        70
     Fault 2       1.00      0.92      0.96        86
     Fault 3       0.86      0.97      0.92        72
     Fault 4       0.74      1.00      0.85        69
     Fault 5       0.96      0.91      0.93        74
     Fault 6       0.98      0.73      0.84        82
     Fault 7       0.98      0.98      0.98        59
     Fault 8       1.00      0.97      0.99        71
    No Fault       1.00      1.00      1.00       702

    accuracy                           0.97      1285
   macro avg       0.95      0.94      0.94      1285
weighted avg       0.97      0.97      0.97      1285

#=======================================================================================================
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, cohen_kappa_score


# Calculate F1-score, Matthews correlation coefficient (MCC), and Cohen's Kappa
f1 = f1_score(decoded_actual_values, decoded_predictions, average='weighted')
mcc = matthews_corrcoef(decoded_actual_values, decoded_predictions)
kappa = cohen_kappa_score(decoded_actual_values, decoded_predictions)

# Display the additional metrics
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")

#=======================================================================================================


