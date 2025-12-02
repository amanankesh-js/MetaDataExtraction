import os
import time
from utils.detect_shots import detect_and_split_scenes
from utils.job_queue import update_job_stage, fetch_next_job, mark_job_failed
from config import LOCAL_VIDEO_DIR, SLEEP_DURATION, DEBUG_MODE, SCENE_THRESHOLD
from utils.aud_db_utils import get_pg_conn


conn = get_pg_conn()
debug = DEBUG_MODE
sleep_time = SLEEP_DURATION

status = "in_progress"
local_base_dir = LOCAL_VIDEO_DIR

i = 0

while True:
    job = fetch_next_job(conn, 'inference', status=status) 
    
    if job:
        try:
            if job["scene_detection_time"] is not None: 
                print("Scene detection already done, skipping...")
                continue
            
            start = time.time()
            s3_key = job['s3_key']
            local_path = job.get('local_path', None)

            if local_path:
                detect_and_split_scenes(video_path=local_path, threshold=SCENE_THRESHOLD)
            else:
                print("\nlocal_path : ", local_path,"\n")
                raise ValueError("Download returned no local_path")

            scene_detection_time = time.time() - start
            
            if local_path:
                update_job_stage(conn, job['id'], 'download', new_status='pending', addons=[f"local_path = '{local_path}'", f"scene_detection_time = {scene_detection_time:0.2f}"])

        except Exception as e:
            print("Download failed for:", s3_key, e)
            mark_job_failed(conn, job['id'])

        if debug:
            print("Exiting due to debug mode")
            break
    else:
        print("Sleeping for 3 mins")
        time.sleep(sleep_time)
