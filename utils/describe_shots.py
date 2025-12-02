import os
import json
import math
import pandas as pd
from google import genai
from datetime import datetime
from google.genai.types import HttpOptions
from concurrent.futures import ThreadPoolExecutor, as_completed


client = genai.Client(
    vertexai=True,
    project="js-titan-dslabs",
    location="us-central1",
    http_options=HttpOptions(api_version="v1")
)

def load_shots(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # If wrapped in {"shots": [...]} or {"scenes": [...]}
    if isinstance(data, dict):
        if "shots" in data:
            data = data["shots"]
        elif "scenes" in data:
            data = data["scenes"]

    shots = []
    for s in data:
        if "start_seconds" in s and "end_seconds" in s:
            start = float(s["start_seconds"])
            end = float(s["end_seconds"])
        else:
            raise ValueError("Shot JSON must contain start_seconds and end_seconds")

        shots.append({
            "start": math.floor(start),
            "end": math.ceil(end)
        })

    return shots



def load_excel_metadata(excel_desc, excel_summary):
    df1 = pd.read_excel(excel_desc)       
    df2 = pd.read_excel(excel_summary)    
    descriptions = df1["description"].tolist()
    summaries = df2["content_summary"].tolist()
    translations = df2["transcript_full_text"].tolist()
    return descriptions, summaries, translations

def build_shot_string(shot_index, start, end, descriptions, summaries, translations):
    text = f"=== SHOT {shot_index} ({start}s → {end}s) ===\n\n"

    for sec in range(start, end + 1):
        # Descriptions = 1 per sec
        desc = descriptions[sec] if sec < len(descriptions) else ""

        # Summary/translation = 1 per 5 sec
        block_index = sec // 5
        summ = summaries[block_index] if block_index < len(summaries) else ""
        trans = translations[block_index] if block_index < len(translations) else ""

        text += (
            f"[FRAME {sec}]\n"
            f"Description: {desc}\n"
            f"Summary (5s block {block_index}): {summ}\n"
            f"transcript_full_text (5s block {block_index}): {trans}\n\n"
        )

    return text


# 4) Build all shot strings
def build_all_shot_strings(shots, descriptions, summaries, translations):
    return [
        build_shot_string(
            i + 1, shot["start"], shot["end"],
            descriptions, summaries, translations
        )
        for i, shot in enumerate(shots)
    ]


# 5) Main function
def generate_shot_wise_text(shots_json, excel_desc, excel_summary, output_file=None):
    shots = load_shots(shots_json)
    descriptions, summaries, translations = load_excel_metadata(excel_desc, excel_summary)
    shot_strings = build_all_shot_strings(shots, descriptions, summaries, translations)

    full = "\n".join(shot_strings)

    if output_file:
        with open(output_file, "w") as f:
            f.write(full)

    return shot_strings, full


def call_gemini(shot_text, prompt_template):
    prompt = prompt_template.replace("{{shots_text}}", shot_text)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


def extract_and_save_json(text: str, output_path: str = None) -> dict:
    """Extract valid JSON from model output."""
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end == -1:
        print("⚠️ No JSON found")
        return {}
    try:
        data = json.loads(text[start:end+1])
    except Exception:
        print("⚠️ Invalid JSON content.")
        data = {}

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON saved: {output_path}")
    return data


def assemble_json_output(shots, gemini_outputs):
    print("Assembling final JSON output...")

    final = []
    for i, (shot, gemini_text) in enumerate(zip(shots, gemini_outputs)):
        print(shot, gemini_text)
        if not gemini_text:
            print(f"⚠️ Empty output for shot {i+1}")
            continue

        start, end = gemini_text.find('{'), gemini_text.rfind('}')
        if start == -1 or end == -1:
            print(f"⚠️ No JSON found in shot {i+1}")
            continue

        try:
            parsed = json.loads(gemini_text[start:end+1])
        except Exception as e:
            print(f"⚠️ JSON parse error in shot {i+1}: {e}")
            continue

        shot_obj = {
            "shot_index": i + 1,
            "start": shot["start"],
            "end": shot["end"]
        }

        if isinstance(parsed, dict):
            for k, v in parsed.items():
                shot_obj[k] = v

        final.append(shot_obj)

    return final



def parallel_infer_gemini(shot_strings, prompt_template, max_workers=10):
    results = [None] * len(shot_strings)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(call_gemini, shot_text, prompt_template): idx
            for idx, shot_text in enumerate(shot_strings)
        }

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
                print(f"✓ Completed shot {idx + 1}")
            except Exception as e:
                print(f"⚠ Error on shot {idx + 1}: {e}")
                results[idx] = None  

    return results


def save_shots_to_excel(final_json, output_excel="shots_gemini_output.xlsx"):
    df = pd.DataFrame(final_json)
    df.to_excel(output_excel, index=False)
    print(f"✓ Saved Excel → {output_excel}")


def make_timestamp_folder(base_dir="output"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(base_dir, timestamp)
    os.makedirs(folder, exist_ok=True)
    return folder


def process_shots(
    output_dir,
    prompt_dir,
    max_workers=8,
):
    prompt_template = f"{prompt_dir}/shots_prompt.txt"
    shots_json=f"{output_dir}/shots.json"
    excel_desc=f"{output_dir}prompt1_prompt2_merged.xlsx"
    excel_summary=f"{output_dir}prompt3_prompt4_merged.xlsx"

    shot_strings, _ = generate_shot_wise_text(
        shots_json=shots_json,
        excel_desc=excel_desc,
        excel_summary=excel_summary,
        output_file=os.path.join(output_dir, "all_shots.txt")
    )
    shots = load_shots(shots_json)
    gemini_outputs = parallel_infer_gemini(shot_strings, prompt_template, max_workers=max_workers)
    final_json = assemble_json_output(shots, gemini_outputs)

    with open(os.path.join(output_dir, "shots_gemini_output.json"), "w") as f:
        json.dump(final_json, f, indent=4, ensure_ascii=False)

    print("\n✓ Saved → shots_gemini_output.json")
    save_shots_to_excel(final_json, os.path.join(output_dir, "shots_gemini_output.xlsx"))


if __name__ == "__main__":
    import time 
    start_time = time.time()
    output_dir = make_timestamp_folder("output")
    shots_json=f"{output_dir}/shots.json"
    excel_desc=f"{output_dir}prompt1_prompt2_merged.xlsx"
    excel_summary=f"{output_dir}prompt3_prompt4_merged.xlsx"
    shots_prompt = "/Users/amana1/working_dir/Meta_Extraction/prompts/shots_prompt.txt"
    output_file=f"{output_dir}/all_shots.txt"
    
    with open(shots_prompt, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    
    process_shots(
        shots_json=shots_json,
        excel_desc=excel_desc,
        excel_summary=excel_summary,
        output_dir=output_dir,
        prompt_template=prompt_template,
        max_workers=8
    )
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n🎉 All done in {total_time//60} mins, {total_time%60} seconds")



"""
character_names: List of character names appearing in the shot.
character_apperances: Descriptions of how each character appears (clothing, expressions, etc.).
character_emotions: Descriptions of the emotions displayed by each character.
shot_summary: A concise summary of the entire shot.
shot_description: A detailed narrative description of the shot, including actions, dialogues, and interactions.
"""