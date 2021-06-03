# %%
from IPython import get_ipython

# %% [markdown]
# Importing the libraries openCV and Matplotlib for reading and plotting images
#

# %%
from tensorflow.keras.datasets import mnist
import numpy as np
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# Downloading the MNIST data

# %%
(x_train, _), (x_test, _) = mnist.load_data()
print("The shape of x_train dataset is", x_train.shape)

# %% [markdown]
# ## Reading greyscale image
# Loading first sample from MNIST dataset. Resizing the image to 18x18.

# %%
# selecting the first sample
x = x_train[1]
print("The dimension of x is 2D matrix as ", x.shape)
# Resizing the image
x = cv2.resize(x, (18, 18))

# %% [markdown]
# Plotting the image using Matplotlib

# %%
plt.imshow(x, cmap='gray')

# %% [markdown]
# You can see that height and width of the matrix is 18x18, same as height and width of above image. So, each pixel is represented by number.

# %%
print("The range of pixel varies between 0 to 255")
print("The pixel having black is more close to 0 and pixel which is white is more close to 255")
print(x)

# %% [markdown]
# ## Reading colour image

# %%
# Reading color image
cat = cv2.imread('cat.png')
plt.imshow(cv2.cvtColor(cat, cv2.COLOR_BGR2RGB))
plt.show()

# %%
print('The shape of image is ', cat.shape)

# %% [markdown]
# ### Plotting the RGB channels of the image.

# %%
cat_r = cv2.imread('cat.png')
cat_r[:, :, 1:2] = 0
plt.imshow(cat_r)


# %%
cat_g = cv2.imread('cat.png')
cat_g[:, :, (0, 2)] = 0
plt.imshow(cat_g)


# %%
cat_b = cv2.imread('cat.png')
cat_b[:, :, 0:1] = 0
plt.imshow(cat_b)
# %%
