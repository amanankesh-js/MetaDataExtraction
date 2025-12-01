import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import insightface

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def l2_normalize(x, axis=-1, eps=1e-10):
    return x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), eps))


def cosine_similarity(a, b):
    a = a.reshape(1, -1)
    return np.sum(a * b, axis=1)


def build_character_db(char_root, face_app):
    character_db = {}
    for char_name in os.listdir(char_root):
        char_dir = os.path.join(char_root, char_name)
        if not os.path.isdir(char_dir):
            continue

        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            img_paths.extend(glob.glob(os.path.join(char_dir, ext)))

        embeddings = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            faces = face_app.get(img)
            if len(faces) > 0:
                embeddings.append(faces[0].normed_embedding)

        if embeddings:
            embeddings = l2_normalize(np.stack(embeddings, axis=0), axis=1)
            character_db[char_name] = {"embeddings": embeddings}

    return character_db


def match_face_to_character(face_emb, character_db, sim_threshold=0.35):
    best_name = "Unknown"
    best_sim = -1
    for cname, data in character_db.items():
        sims = cosine_similarity(face_emb, data["embeddings"])
        max_sim = float(sims.max())
        if max_sim > best_sim:
            best_sim = max_sim
            best_name = cname
    return best_name if best_sim >= sim_threshold else "Unknown"


def annotate_frames(input_frames_dir, output_frames_dir, face_app, character_db):
    ensure_dir(output_frames_dir)

    frame_paths = sorted(
        glob.glob(os.path.join(input_frames_dir, "*.jpg")) +
        glob.glob(os.path.join(input_frames_dir, "*.png")) +
        glob.glob(os.path.join(input_frames_dir, "*.jpeg"))
    )

    for fpath in tqdm(frame_paths, desc="Annotating frames"):
        frame = cv2.imread(fpath)
        faces = face_app.get(frame)

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            name = match_face_to_character(face.normed_embedding, character_db)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out_name = os.path.join(output_frames_dir, os.path.basename(fpath))
        cv2.imwrite(out_name, frame)

    print(f"\n✔ Annotated frames saved at: {output_frames_dir}")


def process_frames(args):
    face_app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    character_db = build_character_db(args['char_dir'], face_app)
    annotate_frames(args['input_dir'], args["out_dir"], face_app, character_db)




def create_cluster_video(video_path, faces, output_path, fps=25):
    """
    Create a debug video with cluster labels drawn on the *original* frames.
    """
    print("🎥 Generating cluster video using original frames...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

    # Group faces by frame index for efficient access
    from collections import defaultdict
    faces_by_frame = defaultdict(list)
    for f in faces:
        faces_by_frame[f["frame_idx"]].append(f)

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="🎞 Rendering cluster video", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # if frame_idx > TEST_FRAMES_LIMIT:
        #     break

        # Draw clusters (use original frame, no resizing)
        if frame_idx in faces_by_frame:
            for f in faces_by_frame[frame_idx]:
                x1, y1, x2, y2 = map(int, f["bbox"])
                cluster = f.get("character", "Unknown")
                color = id_color(cluster)

                # Draw bounding box + cluster label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = str(cluster)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"✅ Cluster video saved: {output_path}")

#
if __name__ == "__main__":

    args = {
        "char_dir": "/Users/amana1/working_dir/zMetaDataExtraction/character_detection/char_dir",
        "input_dir": "/Users/amana1/working_dir/zMetaDataExtraction/output_dir/TX_MASTER_FC_Anupamaa_SH4164_S1_E1599_DYN1492441_v2_763507606_900790931_877219001/frames",
        "out_dir": "/Users/amana1/working_dir/zMetaDataExtraction/output_dir/TX_MASTER_FC_Anupamaa_SH4164_S1_E1599_DYN1492441_v2_763507606_900790931_877219001/frames_annotated",
    }
    process_frames(args)
   
