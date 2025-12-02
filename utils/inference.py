import re
import subprocess
from collections import defaultdict
import json
import os
import tempfile
import time
import glob
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from google.genai.types import HttpOptions


class VideoFrameAudioContextAnalyzer:

    def __init__(self, project_id: str = "js-titan-dslabs", location: str = "us-central1", model: str = "gemini-2.0-flash", temperature: float = 1.5):
        """Initialize Gemini client."""
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            http_options=HttpOptions(api_version="v1"),
        )
        
        self.model = model
        self.temperature = temperature
        self.generate_content_config = types.GenerateContentConfig(
                        temperature = self.temperature,
                        top_p = 0.95,
                        max_output_tokens = 8192,
                        response_modalities = ["TEXT"],
                        safety_settings = [types.SafetySetting(
                          category="HARM_CATEGORY_HATE_SPEECH",
                          threshold="OFF"
                        ),types.SafetySetting(
                          category="HARM_CATEGORY_DANGEROUS_CONTENT",
                          threshold="OFF"
                        ),types.SafetySetting(
                          category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                          threshold="OFF"
                        ),types.SafetySetting(
                          category="HARM_CATEGORY_HARASSMENT",
                          threshold="OFF"
                        )],
                        response_mime_type = "application/json",
                        )


    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Gemini model."""
        try:
            with open(audio_path, "rb") as f:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[{"role": "user", "parts": [
                        {"text": "Please provide an accurate transcript of this audio clip."},
                        {"inline_data": {"mime_type": "audio/wav", "data": f.read()}}
                    ]}], 
                    config=self.generate_content_config
                )
            transcript = response.text.strip()
            print(f"🗣️ Transcribed {os.path.basename(audio_path)}")
            return transcript
        except Exception as e:
            print(f"⚠️ Transcription failed: {e}")
            return ""


    def payload_size(self, parts):
        total = 0
        for p in parts:
            if "text" in p:
                total += len(p["text"].encode("utf-8"))
            elif "inline_data" in p:
                total += len(p["inline_data"]["data"])
        return total


    def analyze_multimodal_segment(self, frame_paths, audio_path, transcript_text, prompt):
        """Send frames + audio + transcript to Gemini."""
        # print(f"🎬 Analyzing {len(frame_paths)} frames + {os.path.basename(audio_path)} with transcript...")
        # print(frame_paths, audio_path)
        parts = [
            {"text": f"{prompt}\n\nHere is the transcript of the segment:\n{transcript_text}"}
        ]
        for frame_path in frame_paths:
            with open(frame_path, "rb") as f:
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": f.read()}})
        with open(audio_path, "rb") as f:
            parts.append({"inline_data": {"mime_type": "audio/wav", "data": f.read()}})
        
        print(f"📦 Payload size: {self.payload_size(parts) / (1024*1024):.2f} MB")

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": parts}],
                config=self.generate_content_config
            )
            return response.text
        except Exception as e:
            print(f"❌ Gemini analysis failed: {e}")
            return ""

    def read_prompts(self, prompts_folder):
        prompt_list = []
        prompt_files = sorted(glob.glob(os.path.join(prompts_folder, 'prompt*.txt')))
        data = {"num_frames": self.chunk_size}
        for file in prompt_files:
            print(file)
            with open(file, 'r') as f:
                prompt = f.read()
                formatted_prompt = prompt.format_map(defaultdict(str, {k: str(v) for k, v in data.items()}))
                print( formatted_prompt)
            prompt_list.append(formatted_prompt)
        return prompt_list
    
    def extract_and_save_json(self, text: str, output_path: str = None) -> dict:
        """Extract valid JSON from model output."""
        start, end = text.find('{'), text.rfind('}')
        if start == -1 or end == -1:
            print("⚠️ No JSON found")
            return {}

        try:
            data = json.loads(text[start:end+1])
        except Exception:
            
            print(f"⚠️ Invalid JSON content: {text}")
            data = {}

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"💾 JSON saved: {output_path}")
        return data


def get_meta_data(args):
    model = args["model"]
    project = args["project"]
    location = args["location"]
    temperature = args["temperature"]
    chunk_size = args["chunk_size"]
    max_workers = args["max_workers"]
    prompt_dir = args["prompt_dir"]
    output = args["output_dir"]

    audio_dir = os.path.join(output, "audio")
    annotated_frames_dir = os.path.join(output, "annotated_frames")
    

    movie_name = os.path.basename(output)

    os.makedirs(output, exist_ok=True)
    print(f"🗂️ Run output directory: {output}")

    analyzer = VideoFrameAudioContextAnalyzer(project_id=project, location=location, model=model, temperature=temperature)
    analyzer.chunk_size=chunk_size
    print("🚀 Starting multi-prompt multimodal video analysis...\n")


    audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    image_files = sorted(glob.glob(os.path.join(annotated_frames_dir, "*.jpg")))
    segments = []

    for i in range(len(audio_files)):
        if (i+1) * chunk_size< len(image_files):
            segments.append((image_files[i * chunk_size: (i+1) * chunk_size], audio_files[i]))
        else:
            segments.append((image_files[i * chunk_size: (i+1) * chunk_size], audio_files[i]))

    prompt_files = analyzer.read_prompts(prompt_dir)
    # prompt_files = sorted(glob.glob(os.path.join(prompt_dir, "prompt*.txt")))
    # if not prompt_files:
    #     print("❌ No prompt files found in folder.")
    #     return
    
    # print("🎤 Transcribing all segments in parallel...", len(segments))
    segments = segments[100:110]
    print("🎤 Transcribing all segments in parallel...", len(segments))
    def transcribe_segment(i, audio_path):
        transcript = analyzer.transcribe_audio(audio_path)   # single .aac
        return i, transcript

    transcripts = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(transcribe_segment, i, audio_path): i
            for i, (_, audio_path) in enumerate(segments)
        }

        for future in as_completed(futures):
            i = futures[future]
            try:
                idx, transcript = future.result()
                transcripts[idx] = transcript
            except Exception as e:
                print(f"⚠️ Transcription failed for segment {i}: {e}")


        for i, prompt in enumerate(prompt_files):
            # prompt_name = os.path.splitext(os.path.basename(prompt_path))[0]
            print(f"\n🧩 Running prompt: prompt{i+1}")
            output_dir = os.path.join(output, "prompt" + str(i+1))
            os.makedirs(output_dir, exist_ok=True)

            def process_segment(i, frames, audio):
                transcript = transcripts.get(i, "")
                if not transcript:
                    return i, {}
                json_text = analyzer.analyze_multimodal_segment(frames, audio, transcript, prompt)
                json_data = analyzer.extract_and_save_json(json_text)   # parse JSON only
                return i, json_data

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_segment, i, frames, audio): i
                    for i, (frames, audio) in enumerate(segments, 0)
                }

                results = {}
                failed = []
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        idx, json_data = future.result()
                        if not json_data:   # {} means Gemini returned invalid JSON
                            failed.append(i)
                        else:
                            results[idx] = json_data
                    except Exception as e:
                        print(f"⚠️ Segment {i} crashed: {e}")
                        failed.append(i)
           
            if failed:
                print(f"🔁 Retrying failed segments: {failed}")
                for i in failed:
                    frames, audio = segments[i]
                    try:
                        transcript = transcripts.get(i, "")
                        json_text = analyzer.analyze_multimodal_segment(frames, audio, transcript, prompt)
                        json_data = analyzer.extract_and_save_json(json_text)
                        results[i] = json_data
                        print(f"✔ Retry success → segment {i}")
                    except Exception as e:
                        print(f"❌ Retry failed → segment {i}: {e}")

            for idx in sorted(results.keys()):
                out_path = os.path.join(output_dir, f"{movie_name}_chunk_{idx:03d}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results[idx], f, ensure_ascii=False, indent=2)
                print("💾 Saved", out_path)

    print("\n🏁 All prompts processed successfully.")


if __name__ == "__main__":
    # import sys, os
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Meta_Extraction.config import *
    output="/Users/amana1/working_dir/zMetaDataExtraction/output_dir/TX_MASTER_FC_Anupamaa_SH4164_S1_E1599_DYN1492441_v2_763507606_900790931_877219001"
    
    args ={
        "output": output,
        "project": PROJECT,
        "location": LOCATION,
        "model": MODEL,
        "audio": f"{output}/audio",
        "frames_annotated": f"{output}/frames",
        "prompt_folder": PROMPT_TEMPLATES_DIR,
        "chunk_size": CHUNK_DURATION,
        "max_workers": 10
    }
    
    get_meta_data(args)
