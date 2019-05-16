import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.utils import class_weight
# import os

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# from keras.applications import VGG16
#Load the VGG model
vgg_conv = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
									
											

train_images = np.load("processed_data/train_total.npy")
train_labels = np.load("processed_data/train_total_label.npy")

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_labels),
                                                 train_labels)

train_labels = keras.utils.to_categorical(train_labels, 58)


# Converts a class vector (integers) to binary class matrix.
# E.g. for use with categorical_crossentropy.
# `to_categorical` converts into a matrix with as many columns as there are classes. The number of rows stays the same.

# print(train_labels.shape)

# train_labels = np.reshape(train_labels, 1, 58)

# train_labels = to_onehot_n(np.load('processed_data/train_total_label.npy').astype(np.int), dim=dim)


# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)



NAME = "VGG-ADAM5-totaltrain-{}LR-0.5-dropout".format(0.0001)

tensorboard = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NAME))
# add early-stopping: stop if there is no improvement on val_loss more than 10 epochs
earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto')
# add reducelronplateau: halve the learning_rate if there is no improvement on val_loss more than 5 epochs
reducelronplateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0, mode='auto')
checkpointfunc = keras.callbacks.ModelCheckpoint('trained_models/{}.model'.format(NAME), monitor='val_loss', verbose=0,
												 save_best_only=True,
												 save_weights_only=False, mode='min', period=1)
# Save the model after every epoch
# save_best_only: if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.
# mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. 
# For val_acc, this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity.
# period: Interval (number of epochs) between checkpoints.


model = keras.Sequential()
model.add(keras.layers.Lambda(lambda x: tf.image.resize_images(x, size=(50, 50))))
# Add the vgg convolutional base model
model.add(vgg_conv)


# Add new layer

# from kaggle
# add one dropout before adding new convolutional layers
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Conv2D(200, kernel_size=(3, 3), padding = "same", activation='relu'))
model.add(keras.layers.Conv2D(180, kernel_size=(3, 3), padding = "same", activation='relu'))
# add one batch normalization layer
model.add(keras.layers.BatchNormalization())
#model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Conv2D(180, kernel_size=(3, 3), padding = "same", activation='relu'))
model.add(keras.layers.Conv2D(140, kernel_size=(3, 3), padding = "same", activation='relu'))
# model.add(keras.layers.Conv2D(100, kernel_size=(3, 3), padding = "same", activation='relu'))
# model.add(keras.layers.Conv2D(50, kernel_size=(3, 3), padding = "same", activation='relu'))
# add one batch normalization layer
model.add(keras.layers.BatchNormalization())
#model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(180, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.5))
# from fine_tune
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(58, activation='softmax'))

#model.summary()

print()
print(NAME)

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# If your targets are one-hot encoded, use categorical_crossentropy. 
# if your targets are integers, use sparse_categorical_crossentropy. 

# load previous trained model for lateron initial_epoch
'''if os.path.exists('trained_models/{}.model'.format(NAME)):
	model = tf.keras.models.load_model('trained_models/{}.model'.format(NAME))
	print("checkpoint_loaded")'''

model.fit(train_images, train_labels, batch_size=100, epochs=35, initial_epoch=0, validation_split=0.20, shuffle=True,
		  callbacks=[tensorboard, checkpointfunc, earlystopping, reducelronplateau], class_weight=class_weights,)
model.save('trained_models/{}.model'.format(NAME))
