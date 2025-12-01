import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import insightface
from sklearn.cluster import DBSCAN
from collections import defaultdict

# ---------------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def id_color(label):
    np.random.seed(abs(hash(label)) % (2**32))
    return tuple(int(x) for x in np.random.randint(0, 255, 3))
# ---------------------------------


def extract_faces(frames_dir, face_app):
    """Detect faces, store bbox + embedding per frame"""
    frame_paths = sorted(
        glob.glob(os.path.join(frames_dir, "*.jpg")) +
        glob.glob(os.path.join(frames_dir, "*.png")) +
        glob.glob(os.path.join(frames_dir, "*.jpeg"))
    )

    all_faces = []
    for idx, fpath in enumerate(tqdm(frame_paths, desc="🔍 Detecting faces")):
        img = cv2.imread(fpath)
        faces = face_app.get(img)

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            all_faces.append({
                "frame_idx": idx,
                "bbox": [x1, y1, x2, y2],
                "embedding": face.normed_embedding
            })
    return all_faces, frame_paths


def cluster_faces(all_faces, eps=0.55, min_samples=4):
    embeddings = np.stack([f["embedding"] for f in all_faces])
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
    raw_labels = clustering.labels_

    # Convert raw DBSCAN labels (-1, 0, 1, 2, ...) to 1-based char_<N> labels
    unique = sorted(list(set(raw_labels)))
    mapping = {}

    next_id = 1
    for lab in unique:
        mapping[lab] = f"char_{next_id}"
        next_id += 1

    for f, raw in zip(all_faces, raw_labels):
        f["cluster"] = mapping[raw]

    print(f"✔ Total clusters formed: {len(unique)}")
    return all_faces


def annotate_frames(all_faces, frame_paths, save_dir):
    """Draw bounding box + cluster label"""
    ensure_dir(save_dir)
    faces_by_frame = defaultdict(list)
    for f in all_faces: faces_by_frame[f["frame_idx"]].append(f)

    for idx, fpath in enumerate(tqdm(frame_paths, desc="🖊 Annotating frames")):
        img = cv2.imread(fpath)

        if idx in faces_by_frame:
            for f in faces_by_frame[idx]:
                x1, y1, x2, y2 = f["bbox"]
                label = f["cluster"]
                color = id_color(label)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imwrite(os.path.join(save_dir, os.path.basename(fpath)), img)

    print(f"📌 Annotated frames saved to: {save_dir}")


def save_cluster_crops(all_faces, frames_dir, save_root):
    """Crop and save faces per cluster"""
    ensure_dir(save_root)
    frame_cache = {}
    frame_paths = sorted(
        glob.glob(os.path.join(frames_dir, "*.jpg")) +
        glob.glob(os.path.join(frames_dir, "*.png")) +
        glob.glob(os.path.join(frames_dir, "*.jpeg"))
    )

    for f in tqdm(all_faces, desc="💾 Saving cropped faces"):
        cname = f["cluster"]
        out_dir = os.path.join(save_root, cname)
        ensure_dir(out_dir)

        frame_idx = f["frame_idx"]
        if frame_idx not in frame_cache:
            frame_cache[frame_idx] = cv2.imread(frame_paths[frame_idx])

        img = frame_cache[frame_idx]
        h, w = img.shape[:2]

        x1, y1, x2, y2 = f["bbox"]

        # clamp bbox to image boundaries
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        crop = img[y1:y2, x1:x2]

        # skip bad crops
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            continue

        count = len(os.listdir(out_dir))
        cv2.imwrite(os.path.join(out_dir, f"{frame_idx}_{count}.jpg"), crop)

    print(f"📁 Clustered faces stored in: {save_root}")


def process_video_faces(out_dir):
    frames_dir = os.path.join(out_dir, "frames")
    annotated_frames_dir = os.path.join(out_dir, "annotated_frames")
    cluster_face_out =  os.path.join(out_dir, "clustered_faces")

    face_app = insightface.app.FaceAnalysis(
        name="buffalo_l", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    all_faces, frame_paths = extract_faces(frames_dir, face_app)
    all_faces = cluster_faces(all_faces)
    annotate_frames(all_faces, frame_paths, annotated_frames_dir)
    save_cluster_crops(all_faces, frames_dir=frames_dir, save_root=cluster_face_out)
    


# --------------------------------- MAIN ---------------------------------
if __name__ == "__main__":
    out_dir = "/Users/amana1/working_dir/Meta_Extraction/media_files/viacom18/movies/hindi/e1599_anupama_None_0"

    # frames_dir = os.path.join(out_dir,"frames")
    # annotated_frames_dir = os.path.join(out_dir, "annotated_frames")
    # cluster_face_out =  os.path.join(out_dir, "clustered_faces")

    # face_app = insightface.app.FaceAnalysis(
    #     name="buffalo_l", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
    # )
    # face_app.prepare(ctx_id=0, det_size=(640, 640))
    # all_faces, frame_paths = extract_faces(frames_dir, face_app)
    # all_faces = cluster_faces(all_faces)
    # annotate_frames(all_faces, frame_paths, annotated_frames_dir)
    # save_cluster_crops(all_faces, frames_dir=frames_dir, save_root=cluster_face_out)

    process_video_faces(out_dir)