####config file example:
# bucket_name: "star-dl-datascience"
# s3_prefix: "projects/contextual_ad/colors_cineplex/hindi/movie_files/"
# download_dir: "media_files/"
# download_log: "metadata/download_log/"
# infer_meta: "metadata/inference_tracker/"

# network: "viacom18"
# media_type: "movies"
# language: "hindi"
# channel: Null
# num_files: 100
# max_size_gb: 4
#---------------------------------

import os
import json
import pandas as pd
from utils import read_yaml
from datetime import datetime
from argparse import ArgumentParser
from utils.vid_db_utils import create_frame_table
from utils.aud_db_utils import get_pg_conn, create_audio_table
from utils.download import list_local_files, generate_new_filename, check_filename

parser = ArgumentParser()
parser.add_argument('--config', type=str, default="/Users/amana1/working_dir/Meta_Extraction/configs/movies_hindi.yaml")
parser.add_argument('--local_dir', type=str, default="/Users/amana1/working_dir/videos")
args = parser.parse_args()


def check_filename(filename):
	"""Check if the filename follows the expected pattern"""
	return True 

conn = get_pg_conn()
config = read_yaml(args.config)

os.makedirs('jobs', exist_ok=True)
priority = 2

# files = list_s3_files(config['bucket_name'], config['s3_prefix'], max_size_gb=config['max_size_gb'], num_movies=config['num_files'])
files = list_local_files(args.local_dir, max_size_gb=config['max_size_gb'], num_movies=config['num_files'])

frame_table = '_'.join([config['network'], config['media_type'], config['language']]) + (('_' + config['channel']) if config['channel'] is not None else '') + '_frame'
print("Using table:", frame_table)
create_frame_table(conn, frame_table)

audio_table = '_'.join([config['network'], config['media_type'], config['language']]) + (('_' + config['channel']) if config['channel'] is not None else '') + '_audio'
print("Using table:", audio_table)
create_audio_table(conn, audio_table)

query = f"""
SELECT DISTINCT movie 
FROM {frame_table};
"""
cursor = conn.cursor()
cursor.execute(query)
results = cursor.fetchall()
existing = pd.DataFrame(results)


df = pd.DataFrame(files, columns=['s3_key'])
df['filename'] = df['s3_key'].apply(lambda x: generate_new_filename(x, config['media_type']))
df['movie'] = df['filename'].str.split('.').str[0]
# print(df['movie'], existing['movie'])

df['check_filename'] = df['filename'].apply(check_filename)
df['config'] = json.dumps(config)
df['stage'] = 'download'
df['priority'] = priority

mask1 = True if existing.empty else (~df['movie'].isin(existing['movie']))
df1 = df[mask1 & (df['check_filename'])]
df2 = df[mask1 & (~ df['check_filename'])]

date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file = os.path.join('jobs', 'job_' + date_time + '.xlsx')
with pd.ExcelWriter(file) as writer:
	df1[['stage', 'priority', 's3_key', 'filename', 'config']].to_excel(writer, sheet_name='enqueue', index=False)
	df2[['s3_key', 'filename', 'config']].to_excel(writer, sheet_name='check', index=False)