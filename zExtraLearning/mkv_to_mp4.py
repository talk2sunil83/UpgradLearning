
# %%
import pandas as pd
import os
import subprocess
from icecream import ic
# from ffmpeg import video
# %%
start_dir = r"E:\OBSRecordings\Videos"

# %%
# ffmpeg -i "E:\OBSRecordings\Videos\2021-06-27 10-52-30.mkv" -codec copy "E:\OBSRecordings\Videos\2021-06-27 10-52-30.mp4" -movflags +faststart


def convert_to_mp4(mkv_file: str, over_write: bool = False) -> bool:
    name, _ = os.path.splitext(mkv_file)
    out_name = name + ".mp4"
    file_exists = os.path.exists(out_name)
    if (not file_exists) or over_write:
        try:
            cmd = f'ffmpeg -i "{mkv_file}" -codec copy "{out_name}" -movflags +faststart'
            cmd = f"{cmd} -y" if over_write and file_exists else cmd
            print(cmd)
            res = subprocess.call(cmd, shell=True)
            if res != 0:
                return False
            return True
        except Exception as ex:
            ic(ex)
            return False

# %%


for path, folder, files in os.walk(start_dir):
    for file in files:
        if file.endswith('.mkv'):
            print(f"Found file: {file}")
            res = convert_to_mp4(os.path.join(start_dir, file), True)
            if res:
                print(f"Finished converting {file}")
            else:
                print("Something wrong")
        else:
            pass
# %%
# %%
pd.DataFrame([range(20)], columns=["Range"])
# %%
