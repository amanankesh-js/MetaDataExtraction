import os
from Meta_Extraction.utils.video_utils import split_video
from Meta_Extraction.config import *
from time import sleep


def get_video_files():
    video_files = []
    return video_files

while True:
    files = get_video_files()
    for file in files:
        video_name = os.path.splitext(os.path.basename(file))[0]
        out_dir = os.path.join(OUTPUT_DIR, video_name)
        split_video(file, out_dir=out_dir, split_duration=CHUNK_DURATION)

    sleep(SLEEP_DURATION)  # Check for new files every 60 seconds