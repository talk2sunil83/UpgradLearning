# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ##### Copyright 2020 The TensorFlow Authors.

# %%
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% [markdown]
# # Introduction to Tensors
# %% [markdown]
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/guide/tensor"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/tensor.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/guide/tensor.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/guide/tensor.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# %%
import tensorflow as tf
import numpy as np

# %% [markdown]
# Tensors are multi-dimensional arrays with a uniform type (called a `dtype`).  You can see all supported `dtypes` at `tf.dtypes.DType`.
# 
# If you're familiar with [NumPy](https://numpy.org/devdocs/user/quickstart.html), tensors are (kind of) like `np.arrays`.
# 
# All tensors are immutable like Python numbers and strings: you can never update the contents of a tensor, only create a new one.
# 
# %% [markdown]
# ## Basics
# 
# Let's create some basic tensors.
# %% [markdown]
# Here is a "scalar" or "rank-0" tensor . A scalar contains a single value, and no "axes".

# %%
# This will be an int32 tensor by default; see "dtypes" below.
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

# %% [markdown]
# A "vector" or "rank-1" tensor is like a list of values. A vector has 1-axis:

# %%
# Let's make this a float tensor.
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)

# %% [markdown]
# A "matrix" or "rank-2" tensor has 2-axes:

# %%
# If you want to be specific, you can set the dtype (see below) at creation time
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)

# %% [markdown]
# <table>
# <tr>
#   <th>A scalar, shape: <code>[]</code></th>
#   <th>A vector, shape: <code>[3]</code></th>
#   <th>A matrix, shape: <code>[3, 2]</code></th>
# </tr>
# <tr>
#   <td>
#    <img src="images/tensor/scalar.png" alt="A scalar, the number 4" />
#   </td>
# 
#   <td>
#    <img src="images/tensor/vector.png" alt="The line with 3 sections, each one containing a number."/>
#   </td>
#   <td>
#    <img src="images/tensor/matrix.png" alt="A 3x2 grid, with each cell containing a number.">
#   </td>
# </tr>
# </table>
# 
# %% [markdown]
# Tensors may have more axes, here is a tensor with 3-axes:

# %%
# There can be an arbitrary number of
# axes (sometimes called "dimensions")
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])

print(rank_3_tensor)

# %% [markdown]
# There are many ways you might visualize a tensor with more than 2-axes.
# %% [markdown]
# <table>
# <tr>
#   <th colspan=3>A 3-axis tensor, shape: <code>[3, 2, 5]</code></th>
# <tr>
# <tr>
#   <td>
#    <img src="images/tensor/3-axis_numpy.png"/>
#   </td>
#   <td>
#    <img src="images/tensor/3-axis_front.png"/>
#   </td>
# 
#   <td>
#    <img src="images/tensor/3-axis_block.png"/>
#   </td>
# </tr>
# 
# </table>
# %% [markdown]
# You can convert a tensor to a NumPy array either using `np.array` or the `tensor.numpy` method:

# %%
np.array(rank_2_tensor)


# %%
rank_2_tensor.numpy()

# %% [markdown]
# Tensors often contain floats and ints, but have many other types, including:
# 
# * complex numbers
# * strings
# 
# The base `tf.Tensor` class requires tensors to be "rectangular"---that is, along each axis, every element is the same size.  However, there are specialized types of tensors that can handle different shapes:
# 
# * Ragged tensors (see [RaggedTensor](#ragged_tensors) below)
# * Sparse tensors (see [SparseTensor](#sparse_tensors) below)
# %% [markdown]
# You can do basic math on tensors, including addition, element-wise multiplication, and matrix multiplication.

# %%
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2])`

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")


# %%
print(a + b, "\n") # element-wise addition
print(a * b, "\n") # element-wise multiplication
print(a @ b, "\n") # matrix multiplication

# %% [markdown]
# Tensors are used in all kinds of operations (ops).

# %%
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# Find the largest value
print(tf.reduce_max(c))
# Find the index of the largest value
print(tf.argmax(c))
# Compute the softmax
print(tf.nn.softmax(c))

# %% [markdown]
# ## About shapes
# %% [markdown]
# Tensors have shapes.  Some vocabulary:
# 
# * **Shape**: The length (number of elements) of each of the dimensions of a tensor.
# * **Rank**: Number of tensor dimensions.  A scalar has rank 0, a vector has rank 1, a matrix is rank 2.
# * **Axis** or **Dimension**: A particular dimension of a tensor.
# * **Size**: The total number of items in the tensor, the product shape vector
# 
# %% [markdown]
# Note: Although you may see reference to a "tensor of two dimensions", a rank-2 tensor does not usually describe a 2D space.
# %% [markdown]
# Tensors and `tf.TensorShape` objects have convenient properties for accessing these:

# %%
rank_4_tensor = tf.zeros([3, 2, 4, 5])

# %% [markdown]
# <table>
# <tr>
#   <th colspan=2>A rank-4 tensor, shape: <code>[3, 2, 4, 5]</code></th>
# </tr>
# <tr>
#   <td>
# <img src="images/tensor/shape.png" alt="A tensor shape is like a vector.">
#     <td>
# <img src="images/tensor/4-axis_block.png" alt="A 4-axis tensor">
#   </td>
#   </tr>
# </table>
# 

# %%
print("Type of every element:", rank_4_tensor.dtype)
print("Number of dimensions:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

# %% [markdown]
# While axes are often referred to by their indices, you should always keep track of the meaning of each. Often axes are ordered from global to local: The batch axis first, followed by spatial dimensions, and features for each location last. This way feature vectors are contiguous regions of memory.
# 
# <table>
# <tr>
# <th>Typical axis order</th>
# </tr>
# <tr>
#     <td>
# <img src="images/tensor/shape2.png" alt="Keep track of what each axis is. A 4-axis tensor might be: Batch, Width, Height, Freatures">
#   </td>
# </tr>
# </table>
# %% [markdown]
# ## Indexing
# %% [markdown]
# ### Single-axis indexing
# 
# TensorFlow follows standard Python indexing rules, similar to [indexing a list or a string in Python](https://docs.python.org/3/tutorial/introduction.html#strings), and the basic rules for NumPy indexing.
# 
# * indexes start at `0`
# * negative indices count backwards from the end
# * colons, `:`, are used for slices: `start:stop:step`
# 

# %%
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())

# %% [markdown]
# Indexing with a scalar removes the dimension:

# %%
print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())

# %% [markdown]
# Indexing with a `:` slice keeps the dimension:

# %%
print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())

# %% [markdown]
# ### Multi-axis indexing
# %% [markdown]
# Higher rank tensors are indexed by passing multiple indices.
# 
# The single-axis exact same rules as in  the single-axis case apply to each axis independently.

# %%
print(rank_2_tensor.numpy())

# %% [markdown]
# Passing an integer for each index the result is a scalar.

# %%
# Pull out a single value from a 2-rank tensor
print(rank_2_tensor[1, 1].numpy())

# %% [markdown]
# You can index using any combination of integers and slices:

# %%
# Get row and column tensors
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")

# %% [markdown]
# Here is an example with a 3-axis tensor:

# %%
print(rank_3_tensor[:, :, 4])

# %% [markdown]
# <table>
# <tr>
# <th colspan=2>Selecting the last feature across all locations in each example in the batch </th>
# </tr>
# <tr>
#     <td>
# <img src="images/tensor/index1.png" alt="A 3x2x5 tensor with all the values at the index-4 of the last axis selected.">
#   </td>
#       <td>
# <img src="images/tensor/index2.png" alt="The selected values packed into a 2-axis tensor.">
#   </td>
# </tr>
# </table>
# %% [markdown]
# ## Manipulating Shapes
# 
# Reshaping a tensor is of great utility. 
# 

# %%
# Shape returns a `TensorShape` object that shows the size on each dimension
x = tf.constant([[1], [2], [3]])
print(x.shape)


# %%
# You can convert this object into a Python list, too
print(x.shape.as_list())

# %% [markdown]
# You can reshape a tensor into a new shape. The `tf.reshape` operation is fast and cheap as the underlying data does not need to be duplicated.

# %%
# You can reshape a tensor to a new shape.
# Note that you're passing in a list
reshaped = tf.reshape(x, [1, 3])


# %%
print(x.shape)
print(reshaped.shape)

# %% [markdown]
# The data maintains its layout in memory and a new tensor is created, with the requested shape, pointing to the same data. TensorFlow uses C-style "row-major" memory ordering, where incrementing the rightmost index corresponds to a single step in memory.

# %%
print(rank_3_tensor)

# %% [markdown]
# If you flatten a tensor you can see what order it is laid out in memory.

# %%
# A `-1` passed in the `shape` argument says "Whatever fits".
print(tf.reshape(rank_3_tensor, [-1]))

# %% [markdown]
# Typically the only reasonable uses of `tf.reshape` are to combine or split adjacent axes (or add/remove `1`s).
# 
# For this 3x2x5 tensor, reshaping to (3x2)x5 or 3x(2x5) are both reasonable things to do, as the slices do not mix:

# %%
print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]))

# %% [markdown]
# <table>
# <th colspan=3>
# Some good reshapes.
# </th>
# <tr>
#   <td>
# <img src="images/tensor/reshape-before.png" alt="A 3x2x5 tensor">
#   </td>
#   <td>
#   <img src="images/tensor/reshape-good1.png" alt="The same data reshaped to (3x2)x5">
#   </td>
#   <td>
# <img src="images/tensor/reshape-good2.png" alt="The same data reshaped to 3x(2x5)">
#   </td>
# </tr>
# </table>
# 
# %% [markdown]
# Reshaping will "work" for any new shape with the same total number of elements, but it will not do anything useful if you do not respect the order of the axes.
# 
# Swapping axes in `tf.reshape` does not work, you need `tf.transpose` for that. 
# 

# %%
# Bad examples: don't do this

# You can't reorder axes with reshape.
print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n") 

# This is a mess
print(tf.reshape(rank_3_tensor, [5, 6]), "\n")

# This doesn't work at all
try:
  tf.reshape(rank_3_tensor, [7, -1])
except Exception as e:
  print(f"{type(e).__name__}: {e}")

# %% [markdown]
# <table>
# <th colspan=3>
# Some bad reshapes.
# </th>
# <tr>
#   <td>
# <img src="images/tensor/reshape-bad.png" alt="You can't reorder axes, use tf.transpose for that">
#   </td>
#   <td>
# <img src="images/tensor/reshape-bad4.png" alt="Anything that mixes the slices of data together is probably wrong.">
#   </td>
#   <td>
# <img src="images/tensor/reshape-bad2.png" alt="The new shape must fit exactly.">
#   </td>
# </tr>
# </table>
# %% [markdown]
# You may run across not-fully-specified shapes. Either the shape contains a `None` (a dimension's length is unknown) or the shape is `None` (the rank of the tensor is unknown).
# 
# Except for [tf.RaggedTensor](#ragged_tensors), such shapes will only occur in the context of TensorFlow's symbolic, graph-building  APIs:
# 
# * [tf.function](function.ipynb) 
# * The [keras functional API](keras/functional.ipynb).
# 
# %% [markdown]
# ## More on `DTypes`
# 
# To inspect a `tf.Tensor`'s data type use the `Tensor.dtype` property.
# 
# When creating a `tf.Tensor` from a Python object you may optionally specify the datatype.
# 
# If you don't, TensorFlow chooses a datatype that can represent your data. TensorFlow converts Python integers to `tf.int32` and Python floating point numbers to `tf.float32`. Otherwise TensorFlow uses the same rules NumPy uses when converting to arrays.
# 
# You can cast from type to type.

# %%
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
# Now, cast to an uint8 and lose the decimal precision
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
print(the_u8_tensor)

# %% [markdown]
# ## Broadcasting
# 
# Broadcasting is a concept borrowed from the [equivalent feature in NumPy](https://numpy.org/doc/stable/user/basics.html).  In short, under certain conditions, smaller tensors are "stretched" automatically to fit larger tensors when running combined operations on them.
# 
# The simplest and most common case is when you attempt to multiply or add a tensor to a scalar.  In that case, the scalar is broadcast to be the same shape as the other argument. 

# %%
x = tf.constant([1, 2, 3])

y = tf.constant(2)
z = tf.constant([2, 2, 2])
# All of these are the same computation
print(tf.multiply(x, 2))
print(x * y)
print(x * z)

# %% [markdown]
# Likewise, 1-sized dimensions can be stretched out to match the other arguments.  Both arguments can be stretched in the same computation.
# 
# In this case a 3x1 matrix is element-wise multiplied by a 1x4 matrix to produce a 3x4 matrix. Note how the leading 1 is optional: The shape of y is `[4]`.

# %%
# These are the same computations
x = tf.reshape(x,[3,1])
y = tf.range(1, 5)
print(x, "\n")
print(y, "\n")
print(tf.multiply(x, y))

# %% [markdown]
# <table>
# <tr>
#   <th>A broadcasted add: a <code>[3, 1]</code> times a <code>[1, 4]</code> gives a <code>[3,4]</code> </th>
# </tr>
# <tr>
#   <td>
# <img src="images/tensor/broadcasting.png" alt="Adding a 3x1 matrix to a 4x1 matrix results in a 3x4 matrix">
#   </td>
# </tr>
# </table>
# 
# %% [markdown]
# Here is the same operation without broadcasting:

# %%
x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3]])

y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])

print(x_stretch * y_stretch)  # Again, operator overloading

# %% [markdown]
# Most of the time, broadcasting is both time and space efficient, as the broadcast operation never materializes the expanded tensors in memory.  
# 
# You see what broadcasting looks like using `tf.broadcast_to`.

# %%
print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))

# %% [markdown]
# Unlike a mathematical op, for example, `broadcast_to` does nothing special to save memory.  Here, you are materializing the tensor.
# 
# It can get even more complicated.  [This section](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html) of Jake VanderPlas's book _Python Data Science Handbook_ shows more broadcasting tricks (again in NumPy).
# %% [markdown]
# ## tf.convert_to_tensor
# 
# Most ops, like `tf.matmul` and `tf.reshape` take arguments of class `tf.Tensor`.  However, you'll notice in the above case, Python objects shaped like tensors are accepted.
# 
# Most, but not all, ops call `convert_to_tensor` on non-tensor arguments.  There is a registry of conversions, and most object classes like NumPy's `ndarray`, `TensorShape`, Python lists, and `tf.Variable` will all convert automatically.
# 
# See `tf.register_tensor_conversion_function` for more details, and if you have your own type you'd like to automatically convert to a tensor.
# %% [markdown]
# ## Ragged Tensors
# 
# A tensor with variable numbers of elements along some axis is called "ragged". Use `tf.ragged.RaggedTensor` for ragged data.
# 
# For example, This cannot be represented as a regular tensor:
# %% [markdown]
# <table>
# <tr>
#   <th>A `tf.RaggedTensor`, shape: <code>[4, None]</code></th>
# </tr>
# <tr>
#   <td>
# <img src="images/tensor/ragged.png" alt="A 2-axis ragged tensor, each row can have a different length.">
#   </td>
# </tr>
# </table>

# %%
ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]


# %%
try:
  tensor = tf.constant(ragged_list)
except Exception as e:
  print(f"{type(e).__name__}: {e}")

# %% [markdown]
# Instead create a `tf.RaggedTensor` using `tf.ragged.constant`:

# %%
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)

# %% [markdown]
# The shape of a `tf.RaggedTensor` contains unknown dimensions:

# %%
print(ragged_tensor.shape)

# %% [markdown]
# ## String tensors
# 
# `tf.string` is a `dtype`, which is to say you can represent data as strings (variable-length byte arrays) in tensors.
# 
# The strings are atomic and cannot be indexed the way Python strings are. The length of the string is not one of the dimensions of the tensor. See `tf.strings` for functions to manipulate them.
# %% [markdown]
# Here is a scalar string tensor:

# %%
# Tensors can be strings, too here is a scalar string.
scalar_string_tensor = tf.constant("Gray wolf")
print(scalar_string_tensor)

# %% [markdown]
# And a vector of strings:
# %% [markdown]
# <table>
# <tr>
#   <th>A vector of strings, shape: <code>[3,]</code></th>
# </tr>
# <tr>
#   <td>
# <img src="images/tensor/strings.png" alt="The string length is not one of the tensor's axes.">
#   </td>
# </tr>
# </table>

# %%
# If you have three string tensors of different lengths, this is OK.
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
# Note that the shape is (3,). The string length is not included.
print(tensor_of_strings)

# %% [markdown]
# In the above printout the `b` prefix indicates that `tf.string` dtype is not a unicode string, but a byte-string. See the [Unicode Tutorial](https://www.tensorflow.org/tutorials/load_data/unicode) for more about working with unicode text in TensorFlow.
# %% [markdown]
# If you pass unicode characters they are utf-8 encoded.

# %%
tf.constant("🥳👍")

# %% [markdown]
# Some basic functions with strings can be found in `tf.strings`, including `tf.strings.split`.

# %%
# You can use split to split a string into a set of tensors
print(tf.strings.split(scalar_string_tensor, sep=" "))


# %%
# ...but it turns into a `RaggedTensor` if you split up a tensor of strings,
# as each string might be split into a different number of parts.
print(tf.strings.split(tensor_of_strings))

# %% [markdown]
# <table>
# <tr>
#   <th>Three strings split, shape: <code>[3, None]</code></th>
# </tr>
# <tr>
#   <td>
# <img src="images/tensor/string-split.png" alt="Splitting multiple strings returns a tf.RaggedTensor">
#   </td>
# </tr>
# </table>
# %% [markdown]
# And `tf.string.to_number`:

# %%
text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))

# %% [markdown]
# Although you can't use `tf.cast` to turn a string tensor into numbers, you can convert it into bytes, and then into numbers.

# %%
byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)


# %%
# Or split it up as unicode and then decode it
unicode_bytes = tf.constant("アヒル 🦆")
unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")

print("\nUnicode bytes:", unicode_bytes)
print("\nUnicode chars:", unicode_char_bytes)
print("\nUnicode values:", unicode_values)

# %% [markdown]
# The `tf.string` dtype is used for all raw bytes data in TensorFlow. The `tf.io` module contains functions for converting data to and from bytes, including decoding images and parsing csv.
# %% [markdown]
# ## Sparse tensors
# 
# Sometimes, your data is sparse, like a very wide embedding space.  TensorFlow supports `tf.sparse.SparseTensor` and related operations to store sparse data efficiently.
# %% [markdown]
# <table>
# <tr>
#   <th>A `tf.SparseTensor`, shape: <code>[3, 4]</code></th>
# </tr>
# <tr>
#   <td>
# <img src="images/tensor/sparse.png" alt="An 3x4 grid, with values in only two of the cells.">
#   </td>
# </tr>
# </table>

# %%
# Sparse tensors store values by index in a memory-efficient manner
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")

# You can convert sparse tensors to dense
print(tf.sparse.to_dense(sparse_tensor))


