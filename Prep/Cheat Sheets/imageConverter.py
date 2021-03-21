# %%
from PIL import Image
import glob
import os
# %%
for file in glob.glob("*.jfif"):
    im = Image.open(file)
    rgb_im = im.convert('RGB')
    rgb_im.save(file.replace("jfif", "png"), quality=100)
    os.remove(file)

# %%
