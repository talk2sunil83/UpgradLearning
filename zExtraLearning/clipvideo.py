# https: // www.geeksforgeeks.org/moviepy-getting-cut-out-of-video-file-clip/
# %%
import os
import subprocess
from datetime import datetime, timedelta
from os import walk
import multiprocessing
import time
# from joblib import Parallel, delayed
# from tqdm import tqdm
# from dask.distributed import Client
# %%

time_format = '%H:%M:%S.%f'
videos_dir = r"F:/Users/SunilYadav/Desktop/Spacy/source"
ffmpeg_dir = r"C:/ffmpeg/bin/"

CORE_COUNT = multiprocessing.cpu_count()

# %%

# Ref: https://www.geeksforgeeks.org/python-program-to-convert-seconds-into-hours-minutes-and-seconds/


def convert_seconds(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%02d:%02d:%02d" % (hour, minutes, seconds)


def command_caller(command=None):
    sp = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    out, err = sp.communicate()
    if sp.returncode:
        print(
            "Return code: %(ret_code)s Error message: %(err_msg)s"
            % {"ret_code": sp.returncode, "err_msg": err}
        )
    return sp.returncode, out.decode("utf-8").strip(), err


# %%
def clip_video(ffmpeg_dir, source_videos_dir, skip, leave, target_videos_dir, clip_name):
    print(f"\t{clip_name}")
    source_file_path = source_videos_dir+"/"+clip_name
    command_duration = f'{ffmpeg_dir}ffprobe.exe -i "{source_file_path}" -show_entries format=duration -v quiet -sexagesimal -of csv=p=0'
    start_time = f"{convert_seconds(skip)}"
    total_duration = datetime.strptime(command_caller(command_duration)[1], time_format)
    end_time = (total_duration - timedelta(seconds=leave)).strftime(time_format)
    # print("\t\t", total_duration, end_time)
    file_name, _ = os.path.splitext(clip_name)
    target_file_name = target_videos_dir+"/"+file_name + ".mp4"
    # if os.path.exists(target_file_name):
    #     os.remove(target_file_name)
    command_clipper = f'{ffmpeg_dir}ffmpeg.exe -y -i "{source_file_path}" -ss "{start_time}" -to "{end_time}" -c:v copy -c:a copy "{target_file_name}"'
    command_caller(command_clipper)


def clip_videos(ffmpeg_dir: str, source_videos_dir: str, skip: int, leave: int, target_videos_dir: str = None):
    """clips and save the videos in mp4 format

    Args:
        ffmpeg_dir (str): ffmpeg dir path
        source_videos_dir (str): source videos directory path
        skip (int): initial skip in seconds
        leave (int): leave of last in seconds
        target_videos_dir (None): target videos directory path. if not provided, source videos directory path will be considered as target
    """
    if target_videos_dir is None:
        target_videos_dir = source_videos_dir
    _, _, clip_names = next(walk(source_videos_dir))
    print("Currently Processing:")

    # futures = []
    # client = Client(n_workers=CORE_COUNT)
    # for clip_name in clip_names:
    #     future = client.submit(clip_video, ffmpeg_dir, source_videos_dir, skip, leave, target_videos_dir, clip_name)
    #     futures.append(future)

    # results = client.gather(futures)
    # client.close()

    for clip_name in clip_names:
        clip_video(ffmpeg_dir, source_videos_dir, skip, leave, target_videos_dir, clip_name)

    print("Processed all files")


clip_videos(ffmpeg_dir, videos_dir, 11, 8)
# %%
