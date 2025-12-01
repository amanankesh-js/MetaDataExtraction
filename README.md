# Meta Data Extraction Pipeline

This repository provides an automated pipeline for extracting structured metadata from long-form video content. It performs video processing, shot segmentation, character recognition, LLM-based inference, and generates JSON + Excel outputs for downstream search, recommendation, and marketing workflows.

## Pipeline Overview

### Step 0 — Video Processing
media_dir → split video → output_dir/<media_name>/frames, audio_chunks, video_chunks

### Step 1 — Shot Detection
media_dir → detect shots → output_dir/<media_name>/shots, output_dir/<media_name>/shots.json

### Step 2 — Character Recognition
frames → annotate faces → output_dir/<media_name>/annotated_frames

### Step 3 — Inference
annotated_frames → LLM inference → output_dir/<media_name>/json/prompt1, output_dir/<media_name>/json/prompt2  
audio_chunks → LLM inference → output_dir/<media_name>/json/prompt3, output_dir/<media_name>/json/prompt4

### Step 4 — Post Processing
prompt1 + prompt2 JSON → Excel → output_dir/<media_name>/excel/prompt1_prompt2.xlsx  
prompt3 + prompt4 JSON → Excel → output_dir/<media_name>/excel/prompt3_prompt4.xlsx

### Step 5 — Shot Description
shots.json + prompt1_prompt2.xlsx → LLM inference → output_dir/<media_name>/shots_description.json

### Step 6 — Scene Description
shots_description.json → scenes_description.json

## Output Directory Structure

output_dir/<media_name>/<time_stamp>/  
    video.mp4  
    frames/  
    audio_chunks/  
    video_chunks/  
    annotated_frames/  
    shots/  
    prompt1/  
    prompt2/  
    prompt3/  
    prompt4/  
    shots.json  
    prompt1_prompt2.xlsx  
    prompt3_prompt4.xlsx  
    shots_description.json  
    shots_description.xlsx  
    scenes_description.json  
    scenes_description.xlsx

## Job Pipeline

create_pipeline_table  
→ create_job_list  
→ enqueue_jobs  
→ download_stage  
→ character_detection  
→ inference_stage  
→ shot_detection  
→ shot_description  
→ scene_detection  
→ scene_description  
→ db_insertion_stage

## Installation

pip install "numpy<2"  
pip install pandas  
pip install opencv-python  
pip install insightface tqdm scikit-learn  
pip install onnxruntime-silicon  
pip install scenedetect  
pip install google-genai  
pip install openpyxl  
pip install pymilvus  
pip install psycopg2-binary  
pip install "sentence-transformers[torch]"  
pip install boto3
