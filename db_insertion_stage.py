from utils.db import get_pg_conn, get_milvus_conn, get_collection, insert_to_db, safe_eval, insert_to_collection, insert_features_to_collection,insert_to_audio_db,insert_to_audio_collection, insert_features_to_audio_collection
from utils.job_queue import *

from sentence_transformers import SentenceTransformer
import pandas as pd

import time

import json

conn = get_pg_conn()
get_milvus_conn()

sleep_time = 2 # 60*3
model = SentenceTransformer("all-mpnet-base-v2")


selected_columns = [
    "audio_events",
    "overall_tone",
    "overall_sentiment",
    "overall_audio_emotion",
    "tone_timestamp_start",
    "tone_timestamp_end",
    "background_type",
    "background_description",
    "background_instruments",
    "background_emotion",
    "song_transcript",
    "song_timestamp_start",
    "song_timestamp_end",
    "song_event",
    "song_placement_flag",
    "brand_utterances"
    "content_summary",
    "speakers",
    "transcript_full_text",
    "translation_approximate"
]

feature_columns = [
    'overall_sentiment', 
    'overall_audio_emotion', 
    'background_instruments',
    'background_emotion',
    'brand_utterances'
]
debug = True
num = 10
i = 0
status = "in_progress"

while True:
    job = fetch_next_job(conn, 'db_insertion', status=status)
    # print(job)
    if job:
        i += 1
        try:
            start = time.time()
            config = job['config']
            table = '_'.join([config['network'], config['media_type'], config['language']]) + (('_' + config['channel']) if config['channel'] is not None else '') + '_audio'
            print(table)
            collection_name = table
            collection = get_collection(collection_name, drop=False)
            print("Using collection:", collection_name)
            processed_output = job['processed_output']

            if not processed_output:
                raise Exception
            
            df = pd.read_excel(processed_output)
        
            insert_to_audio_db(conn, table, df)
            print("Inserted to audio db table:", table)
            insert_to_audio_collection(collection, df, model, feature='content_summary')
            print("Inserted to audio milvus collection:", collection_name)
            insert_features_to_audio_collection(conn, table, model, feature_columns)
            db_insertion_time = time.time() - start
            mark_job_done(conn, job['id'], addons=[f"db_insertion_time = {db_insertion_time:0.2f}"])
        except:
            mark_job_failed(conn, job['id'])

        if debug and i == num:
            print("Exiting due to debug mode")
            break
    else:
        print("Sleeping for 3 mins")
        time.sleep(sleep_time)


