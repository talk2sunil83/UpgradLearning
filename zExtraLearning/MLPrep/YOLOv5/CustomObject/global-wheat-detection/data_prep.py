# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import ast
from os import path
from sklearn.model_selection import train_test_split
from IPython.display import display
from tqdm import tqdm
# %matplotlib inline

# %%
df = pd.read_csv("train.csv")

# %%
df.head()
# %%
df.info()
# %%
df['bbox'] = df['bbox'].apply(ast.literal_eval)
# %%
df.info()
# %%
df.head()
# %%

df = df.groupby("image_id")['bbox'].apply(list).reset_index(name='bboxes')
# %%
df_train, df_test = train_test_split(df, random_state=42, test_size=0.20, shuffle=True)
# %%
df_train.head()
# %%
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
display(df_train.head())
display(df_test.head())
# %%
classes = ["wheat"]

# TODO: Alter it for multiple classes
BASE_PATH = os.getcwd()


def create_dir_structure(base_path=None, base_folder="wheat_data"):
    base_folder = os.path.join(base_path, base_folder) if base_path is not None else base_folder
    os.makedirs(base_folder, exist_ok=True)

    os.makedirs(path.join(base_folder, "images"), exist_ok=True)
    os.makedirs(path.join(base_folder, "images", "train"), exist_ok=True)
    os.makedirs(path.join(base_folder, "images", "validation"), exist_ok=True)
    os.makedirs(path.join(base_folder, "labels"), exist_ok=True)
    os.makedirs(path.join(base_folder, "labels", "train"), exist_ok=True)
    os.makedirs(path.join(base_folder, "labels", "validation"), exist_ok=True)


create_dir_structure(base_path=BASE_PATH)
# %%


# TODO: This function will change if image size is varing, and class is varing
def prepare_data(frame: pd.DataFrame, base_path: str = None, base_folder: str = "wheat_data", data_type: str = 'train', img_w: int = 1024, img_h: int = 1024) -> None:
    base_folder = os.path.join(base_path, base_folder) if base_path is not None else base_folder
    img_w = float(img_w)
    img_h = float(img_h)
    for _, row in tqdm(frame.iterrows(), total=frame.shape[0]):
        image_id = row['image_id']
        bboxes = row['bboxes']
        img_data = []
        # data needs to be class x y w h (Normalized)
        for bbox in bboxes:
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            x_c, y_c, w, h = (x+w/2)/img_w, (y+h/2)/img_h, w/img_w, h/img_h  # Normalized between 0-1
            img_data.append([0, x_c, y_c, w, h])
        img_data = np.array(img_data)
        np.savetxt(
            fname=path.join(base_folder, "labels", data_type, f"{image_id}.txt"),
            X=img_data,
            fmt=["%d", "%f", "%f", "%f", "%f"]
        )
        # TODO: Will change based on data stored
        shutil.copyfile(
            src=f"{base_path}/train/{image_id}.jpg",
            dst=f"{base_folder}/images/{data_type}/{image_id}.jpg")


# %%
prepare_data(df_train, base_path=BASE_PATH)
# %%
prepare_data(df_test, base_path=BASE_PATH, data_type="validation")

# %%
