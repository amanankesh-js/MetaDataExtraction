import os
import time

from utils.describe_shots import process_shots
from utils.job_queue import update_job_stage, fetch_next_job, mark_job_failed
from config import SLEEP_DURATION, DEBUG_MODE,PROMPT_TEMPLATES_DIR, MAX_WORKERS
from utils.aud_db_utils import get_pg_conn

conn = get_pg_conn()
debug = DEBUG_MODE
sleep_time = SLEEP_DURATION
status = "pending"

while True:
    job = fetch_next_job(conn, 'shot_description', status=status) 
    
    if job:
        # try:
        start = time.time()
        
        local_path = job.get('local_path', None)
        if local_path:
            
            dir_path = os.path.dirname(local_path)
            video_name = os.path.splitext(os.path.basename(local_path))[0]
            combined = os.path.join(dir_path, video_name)

            print("\nlocal_path : ", local_path, combined, "\n")
            process_shots(combined, PROMPT_TEMPLATES_DIR, max_workers=MAX_WORKERS)
            shot_description_time = time.time() - start

            update_job_stage(conn, job['id'], 'scene_detection', new_status='pending', addons=[f"local_path = '{local_path}'", f"shot_description_time = {shot_description_time:0.2f}"])
        else:
            mark_job_failed(conn, job['id'])
            raise ValueError("No local_path found in job")
       
        # except Exception as e:
        #     print("Download failed for:", e)

        if debug:
            print("Exiting due to debug mode")
            break
    else:
        print("Sleeping for 3 mins")
        time.sleep(sleep_time)
