import os, json, glob, re
import pandas as pd

def merge_prompt1_prompt2(root, out_name="prompt1_prompt2_merged.xlsx"):
    files1 = sorted(glob.glob(f"{root}/prompt1/*.json"))
    files2 = sorted(glob.glob(f"{root}/prompt2/*.json"))

    rows = {}

    for jf in files1:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                seg = os.path.splitext(os.path.basename(jf))[0]
                rows[seg] = data
        except Exception as e:
            print("❌ Error loading prompt1:", jf, e)

    for jf in files2:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print("❌ Error loading prompt2:", jf, e)
            continue

        seg = os.path.splitext(os.path.basename(jf))[0]
        if seg not in rows:
            continue  # prompt1 missing — skip

        for frame_id, frame_dict in data.items():
            if frame_id not in rows[seg]:
                rows[seg][frame_id] = {}  # create missing frame entry

            for k, v in frame_dict.items():  # add attributes
                rows[seg][frame_id][k] = v


    row_list = []
    for seg_name, frames in rows.items():
        match = re.search(r"chunk_(\d+)", seg_name)
        if match:
            chunk_id = f"chunk_{int(match.group(1)):03d}"
        else:
            raise ValueError(f"Could not extract chunk id from filename: {seg_name}")

        # print("Processing segment:", seg_name, "as", chunk_id, match)
        movie_name = seg_name.replace(chunk_id, "")
        for frame_id, attributes in frames.items():
            entry = dict(attributes)
            entry["movie"] = movie_name
            entry["chunk_id"] = chunk_id
            entry["frame_idx"] = int(frame_id)
            row_list.append(entry)

    df = pd.DataFrame(row_list)
    excel_out = os.path.join(root, out_name)
    df.to_excel(excel_out, index=False)
    print("✔ Excel saved:", excel_out)

    return excel_out


def merge_prompt3_prompt4(root, out_name="prompt3_prompt4_merged.xlsx"):
    files1 = sorted(glob.glob(f"{root}/prompt3/*.json"))
    files2 = sorted(glob.glob(f"{root}/prompt4/*.json"))

    rows = {}

    for jf in files1:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                seg = os.path.splitext(os.path.basename(jf))[0]
                rows[seg] = data
        except Exception as e:
            print("❌ Error loading prompt3:", jf, e)

    for jf in files2:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print("❌ Error loading prompt4:", jf, e)
            continue

        seg = os.path.splitext(os.path.basename(jf))[0]
        if seg not in rows:
            rows[seg] = {}

        for k, v in data.items():
            rows[seg][k] = v

    row_list = []
    for seg_name, meta_dict in rows.items():
        match = re.search(r"chunk_(\d+)", seg_name)
        if match:
            chunk_id = f"chunk_{int(match.group(1)):03d}"
        else:
            raise ValueError(f"Could not extract chunk id from filename: {seg_name}")
        
        # print("Processing segment:", seg_name, "as", chunk_id)
        movie_name = seg_name.replace(chunk_id, "")
        row = dict(meta_dict)
        row["movie"] = movie_name
        row["chunk_id"] = chunk_id
        row_list.append(row)

    df = pd.DataFrame(row_list)
    excel_out = os.path.join(root, out_name)
    df.to_excel(excel_out, index=False)
    print("✔ Excel saved:", excel_out)

    return excel_out


if __name__ == "__main__":
    root = "/Users/amana1/working_dir/zMetaDataExtraction/output/TX_MASTER_FC_Anupamaa_SH4164_S1_E1599_DYN1492441_v2_763507606_900790931_877219001/2025-11-20_15-39-55/video_analysis"
    merge_prompt1_prompt2(root, out_name="prompt1_prompt2_merged.xlsx")
    merge_prompt3_prompt4(root, out_name="prompt3_prompt4_merged.xlsx")