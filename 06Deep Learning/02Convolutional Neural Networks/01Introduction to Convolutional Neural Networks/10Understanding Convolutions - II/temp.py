# %%
from scipy import signal
from scipy import misc
import numpy as np
# %%

ascent = np.array([[1, 4, 0], [0, 9, 1], [5, 0, 7]])
scharr = np.array([[1, -3], [0, 4]])
# grad=signal.convolve2d(ascent, scharr, boundary='symm', mode='same')
grad = signal.convolve2d(ascent, scharr, mode="valid", boundary='wrap')
grad

# %%
1-12+36
# %%
