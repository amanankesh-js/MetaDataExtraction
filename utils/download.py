import boto3
import os
import pandas as pd
import uuid

import yaml
import re

# Allowed media file extensions
MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".mp4", ".mp3", ".wav", ".mov", ".avi"}
s3_client = boto3.client('s3')

def check_filename(text):
    name = text.split('.')[0]
    pattern = r'(?i)^([a-z0-9 ]+(PART \d{1,2})?)_([a-z]{2,3}\d{6,8})_(HD|SD)_(\d{1,4})$'
    match = re.match(pattern, name, re.IGNORECASE)
    if match:
        return True
    else:
        return False
        
def split_by_hd_sd(text):
    match = re.search(r"(.*?)_(HD|SD)(.*)", text, re.IGNORECASE)
    if match:
        left_half = match.group(1).strip()
        keyword = match.group(2)
        right_half = match.group(3).strip()
        return left_half, keyword, right_half
    return text, None, ""

def get_all_fields(text):
    hid = text.split('_')[0]
    res = 'HD' if ('H' in hid) or ('HD' in text) else 'SD'
    
    match = re.search(r"EP[-_]?(\d{1,4})", text, re.IGNORECASE)
    episode = match.group(1)

    match = re.search(rf"{hid}_(.*)[-_\s]EP", text, re.IGNORECASE)
    title = re.sub(r"[-_]", " ", match.group(1))

    match = re.search(r".*(PART[-_\s]\d{1,4})", text)
    if match:
        part = re.sub("-_", " ", match.group(1))
    else:
        part = None

    return hid, title if part is None else ' '.join(title, part), res, episode


def list_s3_files(bucket, prefix, max_size_gb=1.5, num_movies=1000):
    """List all files in an S3 location"""
    print("Listing all files")
    MAX_SIZE_BYTES = max_size_gb * 1024 * 1024 * 1024
    paginator = s3_client.get_paginator("list_objects_v2")
    
    all_objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            all_objects.extend(page['Contents'])

    all_objects.sort(key=lambda x: x['LastModified'], reverse=True)
    
    files = []
    for obj in all_objects:
        key = obj['Key']
        size = obj['Size']
        last_modified = obj['LastModified']

        if size > MAX_SIZE_BYTES:
            print(f"Skipping {key} (size: {size / (1024**3):.2f} GB)")
            continue

        if any(key.lower().endswith(ext) for ext in MEDIA_EXTENSIONS):
            files.append(key)

        if len(files) >= num_movies:
            break
    return files


def generate_new_filename(s3_key, media_type):
    """Generate a new filename using UUID while keeping the original extension"""

    try:
        if media_type == 'gec':
            hid, movie_name, res, episode = get_all_fields(s3_key.split('/')[-1].split('.')[0])
            # hid, movie_name, episode = movie_name_with_hid.split('_')[0], ' '.join(movie_name_with_hid.split('_')[1:-2]), movie_name_with_hid.split('_')[-1]
        elif media_type == 'movies':
            movie_name_with_hid, res, _ = split_by_hd_sd(s3_key.split('/')[-1].split('.')[0])
            hid, movie_name = movie_name_with_hid.split('_')[0], ' '.join(movie_name_with_hid.split('_')[1:])
            episode = None
        else:
            print("Invalid media type in config: Either movies or gec")
            exit()
        ext = os.path.splitext(s3_key)[1]
        return f"{movie_name}_{hid}_{res}_{episode if episode is not None else 0}{ext}"
    except:
        return s3_key.split('/')[-1].split('.')[0]

def download_s3_file(bucket, s3_key, local_dir, new_filename, DOWNLOAD_DIR):
    """Download an S3 file with a new custom name"""
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, new_filename)
    json_dir = local_path.replace(DOWNLOAD_DIR, 'raw_json_outputs/').split('.')[0]
    if os.path.exists(local_path) or os.path.exists(json_dir):
        print("Skipping: ", local_path)
        return None
    s3_client.download_file(bucket, s3_key, local_path)
    print(f"Downloaded: {s3_key} -> {local_path}")
    return local_path




import shutil
from datetime import datetime

def list_local_files(base_dir, max_size_gb=4, num_movies=1000):
    """
    List all allowed media files from a local directory (recursively),
    similar to list_s3_files.
    """
    print("Listing all local files")

    MAX_SIZE_BYTES = max_size_gb * 1024 * 1024 * 1024
    all_files = []

    # Walk recursively through directories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in MEDIA_EXTENSIONS:
                continue

            try:
                size = os.path.getsize(file_path)
                if size > MAX_SIZE_BYTES:
                    print(f"Skipping {file_path} (size: {size / (1024**3):.2f} GB)")
                    continue

                last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                all_files.append({
                    'Key': os.path.relpath(file_path, base_dir),  # similar to S3 key
                    'Size': size,
                    'LastModified': last_modified
                })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Sort by last modified descending
    all_files.sort(key=lambda x: x['LastModified'], reverse=True)

    files = []
    for obj in all_files[:num_movies]:
        files.append(obj['Key'])

    return files

# local_path = download_local_file(local_base_dir, s3_key, local_dir, new_filename, config['download_dir'])
           
def download_local_file(base_dir, local_key, local_dir, new_filename, DOWNLOAD_DIR):    
    """
    "Download" a local file by copying it to a new location with a new name.
    Similar to download_s3_file.
    """
    os.makedirs(local_dir, exist_ok=True)
    source_path = os.path.join(base_dir, local_key)
    local_path = os.path.join(local_dir, new_filename)
    json_dir = local_path.replace(DOWNLOAD_DIR, 'raw_json_outputs/').split('.')[0]
    if os.path.exists(local_path) or os.path.exists(json_dir):
        print("Skipping: ", local_path)
        return None
    try:
        shutil.copy2(source_path, local_path)
        print(f"Copied: {source_path} -> {local_path}")
        return local_path
    except Exception as e:
        print(f"Error copying {source_path} to {local_path}: {e}")
        return None     
