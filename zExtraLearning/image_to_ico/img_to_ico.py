# %%

from PIL import Image
import os

# %%
source_images_dir = "./source"
target_images_dir = "./target"
# %%
_, _, img_names = next(os.walk(source_images_dir))
img_names = [file for file in img_names if file.lower().endswith(('.tiff', '.jpeg', '.jpg', '.png'))]
img_names
# %%
for file_name in img_names:
    file_name = os.path.splitext(file_name)[1]
    target_file_name = f"{target_images_dir}/{file_name}.ico"
    if not os.path.exists(target_file_name):
        img = Image.open(f"{source_images_dir}/{file_name}")
        icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64)]
        img.save(target_file_name, sizes=icon_sizes)
