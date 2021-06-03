# %% [markdown]
# # Gesture Recognition
# In this group project, you are going to build a 3D Conv model that will be able to predict the 5 gestures correctly. Please import the following libraries to get started.

# %%
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf
from keras import backend as K
import random as rn
import datetime
# from scipy.misc import imread, imresize
import os
import numpy as np
import scipy

from imageio import imread
from skimage.transform import resize
scipy.__version__


# %%

# %% [markdown]
# We set the random seed so that the results don't vary drastically.

# %%
random_seed = 30
np.random.seed(random_seed)
rn.seed(random_seed)
tf.random.set_seed(random_seed)

# %% [markdown]
# In this block, you read the folder names for training and validation. You also set the `batch_size` here. Note that you set the batch size in such a way that you are able to use the GPU in full capacity. You keep increasing the batch size until the machine throws an error.
# %%
basepath = os.getcwd() + '\\Project_data'  # pathlib2.Path('./Project_data').resolve()
basepath
# %%
train_doc = np.random.permutation(open(basepath + '\\train.csv').readlines())
val_doc = np.random.permutation(open(basepath + '\\val.csv').readlines())
batch_size = 16  # 32  # 64  # experiment with the batch size

# %% [markdown]
# ## Generator
# This is one of the most important part of the code. The overall structure of the generator has been given. In the generator, you are going to preprocess the images as you have images of 2 different dimensions as well as create a batch of video frames. You have to experiment with `img_idx`, `y`,`z` and normalization such that you get high accuracy.

# %%


def generator_rnn(source_path, folder_list, batch_size):
    print('Source path = ', source_path, '; batch size =', batch_size)
    img_idx = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]  # create a list of image numbers you want to use for a particular video
    while True:
        t = np.random.permutation(folder_list)
        num_batches = int(len(t)/batch_size)  # calculate the number of batches
        for batch in range(num_batches):  # we iterate over the number of batches
            batch_data = np.zeros((batch_size, 15, 120, 120, 3))  # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
            batch_labels = np.zeros((batch_size, 5))  # batch_labels is the one hot representation of the output
            for folder in range(batch_size):  # iterate over the batch_size
                imgs = os.listdir(source_path+'/' + t[folder + (batch*batch_size)].split(';')[0])  # read all the images in the folder
                for idx, item in enumerate(img_idx):  # Iterate over the frames/images of a folder to read them in
                    image = imread(source_path+'/' + t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)

                    if image.shape[1] == 160:
                        image = image[:, 20:140, :].astype(np.float32)
                    else:
                        image = resize(image, (120, 120)).astype(np.float32)

                    # crop the images and resize them. Note that the images are of 2 different shape
                    # and the conv3D will throw error if the inputs in a batch have different shapes

                    batch_data[folder, idx, :, :, 0] = image[:, :, 0] - 104  # normalise and feed in the image
                    batch_data[folder, idx, :, :, 1] = image[:, :, 1] - 117  # normalise and feed in the image
                    batch_data[folder, idx, :, :, 2] = image[:, :, 2] - 123  # normalise and feed in the image

                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels  # you yield the batch_data and the batch_labels, remember what does yield do

        # write the code for the remaining data points which are left after full batches
        if (len(t) % batch_size) != 0:
            batch_data = np.zeros((len(t) % batch_size, 15, 120, 120, 3))
            batch_labels = np.zeros((len(t) % batch_size, 5))
            for folder in range(len(t) % batch_size):
                imgs = os.listdir(source_path+'/' + t[folder + (num_batches*batch_size)].split(';')[0])
                for idx, item in enumerate(img_idx):
                    image = imread(source_path+'/' + t[folder + (num_batches*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                    if image.shape[1] == 160:
                        image = image[:, 20:140, :].astype(np.float32)
                    else:
                        image = resize(image, (120, 120)).astype(np.float32)

                    batch_data[folder, idx, :, :, 0] = image[:, :, 0] - 104
                    batch_data[folder, idx, :, :, 1] = image[:, :, 1] - 117
                    batch_data[folder, idx, :, :, 2] = image[:, :, 2] - 123

                batch_labels[folder, int(t[folder + (num_batches*batch_size)].strip().split(';')[2])] = 1

            yield batch_data, batch_labels
# %% [markdown]
# Note here that a video is represented above in the generator as (number of images, height, width, number of channels). Take this into consideration while creating the model architecture.


# %%
curr_dt_time = datetime.datetime.now()
train_path = basepath + '\\train'
val_path = basepath + '\\val'
num_train_sequences = len(train_doc)
print('# training sequences =', num_train_sequences)
num_val_sequences = len(val_doc)
print('# validation sequences =', num_val_sequences)
num_epochs = 30  # choose the number of epochs
print('# epochs =', num_epochs)

# %% [markdown]
# ## Model
# Here you make the model using different functionalities that Keras provides. Using RNN

# %%

# write your model here

base_model = VGG16(include_top=False, weights='imagenet', input_shape=(120, 120, 3))
x = base_model.output
x = Flatten()(x)
# x.add(Dropout(0.5))
features = Dense(64, activation='relu')(x)
conv_model = Model(inputs=base_model.input, outputs=features)

for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(TimeDistributed(conv_model, input_shape=(15, 120, 120, 3)))
model.add(GRU(32, return_sequences=True))
model.add(GRU(16))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='softmax'))

# %% [markdown]
# Now that you have written the model, the next step is to `compile` the model. When you print the `summary` of the model, you'll see the total number of parameters you have to train.

# %%
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print(model.summary())

# %% [markdown]
# Let us create the `train_generator` and the `val_generator` which will be used in `.fit_generator`.

# %%
train_generator = generator_rnn(train_path, train_doc, batch_size)
val_generator = generator_rnn(val_path, val_doc, batch_size)


# %%
model_name = 'model_init' + '_' + str(curr_dt_time).replace(' ', '').replace(':', '_') + '/'

if not os.path.exists(model_name):
    os.mkdir(model_name)

filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min', epsilon=0.0001, cooldown=0, min_lr=0.00001)  # write the REducelronplateau code here
callbacks_list = [checkpoint, LR]

# %% [markdown]
# The `steps_per_epoch` and `validation_steps` are used by `fit_generator` to decide the number of next() calls it need to make.

# %%
if (num_train_sequences % batch_size) == 0:
    steps_per_epoch = int(num_train_sequences/batch_size)
else:
    steps_per_epoch = (num_train_sequences//batch_size) + 1

if (num_val_sequences % batch_size) == 0:
    validation_steps = int(num_val_sequences/batch_size)
else:
    validation_steps = (num_val_sequences//batch_size) + 1

# %% [markdown]
# Let us now fit the model. This will start training the model and with the help of the checkpoints, you'll be able to save the model at the end of each epoch.

# %%
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1,
                    callbacks=callbacks_list, validation_data=val_generator,
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)

# %%
# Generator for Cove 3d


def generator(source_path, folder_list, batch_size):
    print('Source path = ', source_path, '; batch size =', batch_size)
    img_idx = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27, 28, 29]  # create a list of image numbers you want to use for a particular video
    while True:
        t = np.random.permutation(folder_list)
        num_batches = int(len(t)/batch_size)  # calculate the number of batches
        for batch in range(num_batches):  # we iterate over the number of batches
            batch_data = np.zeros((batch_size, 18, 84, 84, 3))  # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
            batch_labels = np.zeros((batch_size, 5))  # batch_labels is the one hot representation of the output
            for folder in range(batch_size):  # iterate over the batch_size
                imgs = os.listdir(source_path+'/' + t[folder + (batch*batch_size)].split(';')[0])  # read all the images in the folder
                for idx, item in enumerate(img_idx):  # Iterate over the frames/images of a folder to read them in
                    image = imread(source_path+'/' + t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)

                    if image.shape[1] == 160:
                        image = resize(image[:, 20:140, :], (84, 84)).astype(np.float32)
                    else:
                        image = resize(image, (84, 84)).astype(np.float32)

                    # crop the images and resize them. Note that the images are of 2 different shape
                    # and the conv3D will throw error if the inputs in a batch have different shapes

                    batch_data[folder, idx, :, :, 0] = image[:, :, 0] - 104  # normalise and feed in the image
                    batch_data[folder, idx, :, :, 1] = image[:, :, 1] - 117  # normalise and feed in the image
                    batch_data[folder, idx, :, :, 2] = image[:, :, 2] - 123  # normalise and feed in the image

                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels  # you yield the batch_data and the batch_labels, remember what does yield do

        # write the code for the remaining data points which are left after full batches
        if (len(t) % batch_size) != 0:
            batch_data = np.zeros((len(t) % batch_size, 18, 84, 84, 3))
            batch_labels = np.zeros((len(t) % batch_size, 5))
            for folder in range(len(t) % batch_size):
                imgs = os.listdir(source_path+'/' + t[folder + (num_batches*batch_size)].split(';')[0])
                for idx, item in enumerate(img_idx):
                    image = imread(source_path+'/' + t[folder + (num_batches*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                    if image.shape[1] == 160:
                        image = resize(image[:, 20:140, :], (84, 84)).astype(np.float32)
                    else:
                        image = resize(image, (84, 84)).astype(np.float32)

                    batch_data[folder, idx, :, :, 0] = image[:, :, 0] - 104
                    batch_data[folder, idx, :, :, 1] = image[:, :, 1] - 117
                    batch_data[folder, idx, :, :, 2] = image[:, :, 2] - 123

                batch_labels[folder, int(t[folder + (num_batches*batch_size)].strip().split(';')[2])] = 1

            yield batch_data, batch_labels


# %% [markdown]
'''
## Model
Here I will make the model using different functionalities that Keras provides. Remember to use `Conv3D` and `MaxPooling3D` and not `Conv2D` and `Maxpooling2D`. Also remember that the last layer is the softmax. Remember that the network is designed in such a way that the model is able to fit in the memory of the webcam.
'''
# %%
model = Sequential()
model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', input_shape=(18, 84, 84, 3)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1)))

model.add(Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

# model.add(Dropout(0.25))

model.add(Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

# model.add(Dropout(0.25))

model.add(Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

# %% [markdown]
'''
Now that you have written the model, the next step is to `compile` the model. When you print the `summary` of the model, you'll see the total number of parameters you have to train.
'''
# %%
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print(model.summary())
# %% [markdown]
'''
Let us create the `train_generator` and the `val_generator` which will be used in `.fit_generator`.
'''

# %%
train_generator = generator(train_path, train_doc, batch_size)
val_generator = generator(val_path, val_doc, batch_size)
# %% [markdown]
'''

'''

# %%
model_name = 'model_conv3d_init' + '_' + str(curr_dt_time).replace(' ', '').replace(':', '_') + '/'

if not os.path.exists(model_name):
    os.mkdir(model_name)

filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min', epsilon=0.0001, cooldown=0, min_lr=0.00001)
callbacks_list = [checkpoint, LR]

# %%
if (num_train_sequences % batch_size) == 0:
    steps_per_epoch = int(num_train_sequences/batch_size)
else:
    steps_per_epoch = (num_train_sequences//batch_size) + 1

if (num_val_sequences % batch_size) == 0:
    validation_steps = int(num_val_sequences/batch_size)
else:
    validation_steps = (num_val_sequences//batch_size) + 1

# %% [markdown]
'''
Let us now fit the model. This will start training the model and with the help of the checkpoints, you'll be able to save the model at the end of each epoch.
'''
# %%
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1,
                    callbacks=callbacks_list, validation_data=val_generator,
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)

# %%
