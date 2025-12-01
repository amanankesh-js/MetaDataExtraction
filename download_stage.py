import os
import time
from utils.download import download_s3_file, download_local_file

from utils.job_queue import update_job_stage, fetch_next_job, mark_job_failed
from config import LOCAL_VIDEO_DIR, SLEEP_DURATION, DEBUG_MODE
from utils.aud_db_utils import get_pg_conn
from utils.video_utils import split_video

conn = get_pg_conn()
debug = DEBUG_MODE
sleep_time = SLEEP_DURATION

status = "failed"
local_base_dir = LOCAL_VIDEO_DIR

i = 0
while True:
    job = fetch_next_job(conn, 'download', status=status) 
    
    if job:
        try:
            start = time.time()
            s3_key = job['s3_key']
            new_filename = job['filename'].replace('/', '\/')
            config = job['config']
            print("config : ", new_filename, "\n")

            local_dir = os.path.join(config['download_dir'], config['network'], config['media_type'], config['language'], config['channel'] if config['channel'] is not None else '')
            # local_path = download_s3_file('star-dl-datascience', s3_key, local_dir, new_filename, config['download_dir'])
            
            print(s3_key, local_dir, new_filename, config['download_dir'])
            local_path = download_local_file(local_base_dir, s3_key, local_dir, new_filename, config['download_dir'])
            if local_path:
                split_video(local_path, local_dir, split_duration=5)
            else:
                print("\nlocal_path : ", local_path,"\n")

            download_time = time.time() - start
            if local_path:
                print("Downloaded to:", local_path)
                update_job_stage(conn, job['id'], 'inference', new_status='pending', addons=[f"local_path = '{local_path}'", f"download_time = {download_time:0.2f}"])
       
        except Exception as e:
            print("Download failed for:", s3_key, e)
            mark_job_failed(conn, job['id'])

        if debug:
            print("Exiting due to debug mode")
            break
        
    else:
        print("Sleeping for 3 mins")
        time.sleep(sleep_time)

# split_video(video_path, out_dir, split_duration=5, resize_w=640, resize_h=640):