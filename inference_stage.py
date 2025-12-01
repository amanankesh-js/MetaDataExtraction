from utils.inference import GeminiAudioInfer, restore_terminal
from utils.db import get_pg_conn
from utils.job_queue import *

import json
import time
import os

conn = get_pg_conn()
sleep_time = 2 #60 * 3

chunk_size = 5        # seconds per audio chunk
temperature = 0.5
max_workers = 10
debug = False
status = "pending"
i = 0

while True:
    job = fetch_next_job(conn, 'inference', status=status) # if fetched then status="in_progress"

    if job:
    # try:
        start = time.time()
        audio_file = job['local_path']
        movie_name = audio_file.split('/')[-1].split('.')[0]
        config = job['config']

        output_dir = os.path.join(
            'raw_json_outputs',
            config['network'],
            config['media_type'],
            config['language'],
            config['channel'] if config['channel'] is not None else '',
            movie_name
        )

        print(audio_file)
        infer = GeminiAudioInfer(
            audio_file,
            output_dir,
            chunk_size=chunk_size,
            temperature=temperature,
            max_workers=max_workers
        )

        # Skip if already processed
        if os.path.exists(infer.json_dir):
            print("Skipping ...")
            update_job_stage(conn, job['id'], 'db_insertion', new_status="failed")
            continue

        # Check audio metadata
        metadata, meta_flag = infer.check_audio()
        if not meta_flag:
            mark_job_failed(conn, job['id'], addons=[f"metadata = '{json.dumps(metadata)}'::jsonb"])
            continue

        # Run inference
        prompt_list = infer.read_prompts('prompts/')
        audio_chunks = infer.read_all_audio_chunks()
        print("Starting inference")

        infer_logs = infer.infer_all_audio(prompt_list[-2:], audio_chunks)      
        processed_output = infer.process_audio_to_excel()
        inference_time = time.time() - start
        print("Inference time:", inference_time)

        update_job_stage(
            conn,
            job['id'],
            'db_insertion',
            new_status='pending',
            addons=[
                f"metadata = '{json.dumps(metadata)}'::jsonb",
                f"processed_output = '{processed_output}'",
                f"infer_logs = '{json.dumps(infer_logs)}'::jsonb",
                f"inference_time = {inference_time:0.2f}"
            ]
        )

    # except KeyboardInterrupt:
    #     update_job_stage(conn, job['id'], 'inference')
    #     break

    # except Exception as e:
    #     print(f"Job failed: {str(e)}")
    #     mark_job_failed(conn, job['id'])

        if debug:
            restore_terminal()
            print("Exiting due to debug mode")
            break

    else:
        print(f"Sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)
