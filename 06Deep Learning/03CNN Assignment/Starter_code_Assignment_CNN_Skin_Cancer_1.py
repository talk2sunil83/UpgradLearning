# %% [markdown]
# Problem statement: To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.
# %% [markdown]
# ### Importing Skin Cancer Data
# #### To do: Take necessary actions to read the data
# %% [markdown]
# ### Importing all the important libraries

# %%
import Augmentor
from IPython import get_ipython
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import sys
from pathlib2 import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
import kerastuner as kt
from kerastuner import HyperModel, RandomSearch, Hyperband, BayesianOptimization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
# %%
# Ref: https://www.tensorflow.org/tutorials/images/classification

# %% [markdown]
# This assignment uses a dataset of about 2357 images of skin cancer types. The dataset contains 9 sub-directories in each train and test subdirectories. The 9 sub-directories contains the images of 9 skin cancer types respectively.

# %%
root_folder = Path().resolve()
images_path = root_folder / "Skin cancer ISIC The International Skin Imaging Collaboration"
images_path


# %%
# Defining the path for train and test images
data_dir_train = images_path / "Train"
data_dir_test = images_path / "Test"


# %%
train_image_file_names = list(data_dir_train.glob('*/*.jpg'))
image_count_train = len(train_image_file_names)
print(image_count_train)
test_image_file_names = list(data_dir_test.glob('*/*.jpg'))
image_count_test = len(test_image_file_names)
print(image_count_test)

# %% [markdown]
# ### Load using keras.preprocessing
#
# Let's load these images off disk using the helpful image_dataset_from_directory utility.
# %% [markdown]
# ### Create a dataset
#
# Define some parameters for the loader:

# %%
batch_size = 32
img_height = 180
img_width = 180
SEED = 123
image_size = (img_height, img_width)

# %% [markdown]
# Use 80% of the images for training, and 20% for validation.

# %%
# Write your train dataset here
# Note use seed=123 while creating your dataset using tf.keras.preprocessing.image_dataset_from_directory
# Note, make sure your resize your images to the size img_height*img_width, while writting the dataset
train_ds = image_dataset_from_directory(data_dir_train, seed=SEED, image_size=image_size, batch_size=batch_size, validation_split=0.2, subset="training")
train_ds

# %%
# Write your validation dataset here
# Note use seed=123 while creating your dataset using tf.keras.preprocessing.image_dataset_from_directory
# Note, make sure your resize your images to the size img_height*img_width, while writting the dataset
val_ds = image_dataset_from_directory(data_dir_test, seed=SEED, image_size=image_size, batch_size=batch_size, subset="validation", validation_split=0.2)


# %%
# List out all the classes of skin cancer and store them in a list.
# You can find the class names in the class_names attribute on these datasets.
# These correspond to the directory names in alphabetical order.
class_names = train_ds.class_names
print(class_names)

# %% [markdown]
# ### Visualize the data
# %%
fig = plt.figure(figsize=(10, 10))
for i in range(len(class_names)):
    ax = plt.subplot(3, 3, i + 1)
    class_name = class_names[i]
    img_dir = data_dir_train / class_name
    img_num = np.random.randint(image_count_train)
    im = plt.imread(str(train_image_file_names[img_num]))
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()

# %%
# train_ds_np =
# train_images = train_ds.reshape(len(train_ds), img_height, img_width, 3)
# test_images = val_ds.reshape(len(val_ds), img_height, img_width, 3)

# %% [markdown]
# The `image_batch` is a tensor of the shape `(32, 180, 180, 3)`. This is a batch of 32 images of shape `180x180x3` (the last dimension refers to color channels RGB). The `label_batch` is a tensor of the shape `(32,)`, these are corresponding labels to the 32 images.
# %% [markdown]
# `Dataset.cache()` keeps the images in memory after they're loaded off disk during the first epoch.
#
# `Dataset.prefetch()` overlaps data preprocessing and model execution while training.
# %%
# for image_batch, labels_batch in train_ds:
#     print(image_batch.shape)
#     print(labels_batch.shape)
#     break
# %%
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# %% [markdown]
# ### Create the model
# #### Todo: Create a CNN model, which can accurately detect 9 classes present in the dataset. Use ```layers.experimental.preprocessing.Rescaling``` to normalize pixel values between (0,1). The RGB channel values are in the `[0, 255]` range. This is not ideal for a neural network. Here, it is good to standardize values to be in the `[0, 1]`
# %%
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)
augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))


# %%
# Building Hyperparameter tuner
# Note: Scaling layer and model compilation is part of ClassificationHyperModel


# %%

# %% [markdown]
# ### Compile the model
# Choose an appropriate optimizer and loss function for model training


# %%
# Todo, choose an appropriate optimizer and loss function
#  Optimizer and loss function is part of ClassificationHyperModel
# %% [markdown]
'''
#### Build, Run and Evaluate Random Search Tuner
'''
# %%


# # %%
# for image_batch, labels_batch in train_ds:
#     print(image_batch.shape)
#     print(labels_batch.shape)
#     break

# %%
#  Extract X and Y
# images, labels = tuple(zip(*train_ds))
# %%


# %%
# View the summary of all layers
model.summary()

# %% [markdown]
# ### Train the model

# %%
epochs = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# %% [markdown]
# ### Visualizing training results

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %% [markdown]
# #### Todo: Write your findings after the model fit, see if there is an evidence of model overfit or underfit
# %% [markdown]
# ### Write your findings here

# %%
# Todo, after you have analysed the model fit history for presence of underfit or overfit, choose an appropriate data augumentation strategy.
# Your code goes here


# %%
# Todo, visualize how your augmentation strategy works for one instance of training image.
# Your code goes here

# %% [markdown]
# ### Todo:
# ### Create the model, compile and train the model
#

# %%
# You can use Dropout layer if there is an evidence of overfitting in your findings

# Your code goes here

# %% [markdown]
# ### Compiling the model

# %%
# Your code goes here

# %% [markdown]
# ### Training the model

# %%
# Your code goes here, note: train your model for 20 epochs
history =  # your training code

# %% [markdown]
# ### Visualizing the results

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %% [markdown]
# #### Todo: Write your findings after the model fit, see if there is an evidence of model overfit or underfit. Do you think there is some improvement now as compared to the previous model run?
# %% [markdown]
# #### **Todo:** Find the distribution of classes in the training dataset.
# #### **Context:** Many times real life datasets can have class imbalance, one class can have proportionately higher number of samples compared to the others. Class imbalance can have a detrimental effect on the final model quality. Hence as a sanity check it becomes important to check what is the distribution of classes in the data.

# %%
# Your code goes here.

# %% [markdown]
# #### **Todo:** Write your findings here:
# #### - Which class has the least number of samples?
# #### - Which classes dominate the data in terms proportionate number of samples?
#
# %% [markdown]
# #### **Todo:** Rectify the class imbalance
# #### **Context:** You can use a python package known as `Augmentor` (https://augmentor.readthedocs.io/en/master/) to add more samples across all classes so that none of the classes have very few samples.

# %%
get_ipython().system('pip install Augmentor')

# %% [markdown]
# To use `Augmentor`, the following general procedure is followed:
#
# 1. Instantiate a `Pipeline` object pointing to a directory containing your initial image data set.<br>
# 2. Define a number of operations to perform on this data set using your `Pipeline` object.<br>
# 3. Execute these operations by calling the `Pipeline’s` `sample()` method.
#

# %%
path_to_training_dataset = "To do"
for i in class_names:
    p = Augmentor.Pipeline(path_to_training_dataset + i)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.sample(500)  # We are adding 500 samples per class to make sure that none of the classes are sparse.

# %% [markdown]
# Augmentor has stored the augmented images in the output sub-directory of each of the sub-directories of skin cancer types.. Lets take a look at total count of augmented images.

# %%
image_count_train = len(list(data_dir_train.glob('*/output/*.jpg')))
print(image_count_train)

# %% [markdown]
# ### Lets see the distribution of augmented data after adding new images to the original training data.

# %%
path_list = [x for x in glob(os.path.join(data_dir_train, '*', 'output', '*.jpg'))]
path_list


# %%
lesion_list_new = [os.path.basename(os.path.dirname(os.path.dirname(y))) for y in glob(os.path.join(data_dir_train, '*', 'output', '*.jpg'))]
lesion_list_new


# %%
dataframe_dict_new = dict(zip(path_list_new, lesion_list_new))


# %%
df2 = pd.DataFrame(list(dataframe_dict_new.items()), columns=['Path', 'Label'])
new_df = original_df.append(df2)


# %%
new_df['Label'].value_counts()

# %% [markdown]
# So, now we have added 500 images to all the classes to maintain some class balance. We can add more images as we want to improve training process.
# %% [markdown]
# #### **Todo**: Train the model on the data created using Augmentor

# %%
batch_size = 32
img_height = 180
img_width = 180

# %% [markdown]
# #### **Todo:** Create a training dataset

# %%
data_dir_train = "path to directory with training data + data created using augmentor"
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    seed=123,
    validation_split=0.2,
    subset=# Todo choose the correct parameter value, so that only training data is refered to,,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# %% [markdown]
# #### **Todo:** Create a validation dataset

# %%
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    seed=123,
    validation_split=0.2,
    subset=# Todo choose the correct parameter value, so that only validation data is refered to,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# %% [markdown]
# #### **Todo:** Create your model (make sure to include normalization)

# %%
# your code goes here

# %% [markdown]
# #### **Todo:** Compile your model (Choose optimizer and loss function appropriately)

# %%
# your code goes here

# %% [markdown]
# #### **Todo:**  Train your model

# %%
epochs = 30
# Your code goes here, use 50 epochs.
history =  # your model fit code

# %% [markdown]
# #### **Todo:**  Visualize the model results

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %% [markdown]
# #### **Todo:**  Analyze your results here. Did you get rid of underfitting/overfitting? Did class rebalance help?
#
#

# %%
