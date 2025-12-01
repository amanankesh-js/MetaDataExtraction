import os
import cv2
import subprocess
import math


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def split_video(video_path, out_dir, split_duration=5, resize_w=640):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    base_dir = os.path.join(out_dir, video_name)

    frames_dir = os.path.join(base_dir, "frames")
    # video_dir  = os.path.join(base_dir, "video")
    audio_dir  = os.path.join(base_dir, "audio")

    ensure_dir(frames_dir)
    # ensure_dir(video_dir)
    ensure_dir(audio_dir)

    duration = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]).decode().strip())
    num_chunks = math.ceil(duration / split_duration)

    print(f"Total duration: {duration:.2f}s | Chunks: {num_chunks}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    for i in range(num_chunks):
        start = i * split_duration

        # chunk_video = os.path.join(video_dir, f"{video_name}_chunk_{i:04d}.mp4")
        # subprocess.run([
        #     "ffmpeg", "-y",
        #     "-ss", str(start),
        #     "-t", str(split_duration),
        #     "-i", video_path,
        #     "-vf", f"scale={resize_w}:-1",
        #     "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        #     "-c:a", "aac",
        #     chunk_video
        # ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        chunk_audio = os.path.join(audio_dir, f"{video_name}_chunk_{i:04d}.wav")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path, "-ss", str(start),
            "-t", str(split_duration),
            "-vn", "-acodec", "pcm_s16le",
            chunk_audio
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        for sec in range(split_duration):
            frame_time = start + sec
            frame_index = int(frame_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                new_h = int(h * (resize_w / w))
                resized_frame = cv2.resize(frame, (resize_w, new_h), interpolation=cv2.INTER_LINEAR)
                frame_count += 1
                cv2.imwrite(os.path.join(frames_dir, f"{frame_count:08d}.jpg"), resized_frame)

        print(f"✔ Chunk {i+1}/{num_chunks} processed")

    cap.release()
    print("\n🎉 Finished!")


def split_video_files(files, output_dir, chunk_duration=5):
    for file in files:
        video_name = os.path.splitext(os.path.basename(file))[0]
        out_dir = os.path.join(output_dir, video_name)
        split_video(file, out_dir=out_dir, split_duration=chunk_duration)

if __name__ == "__main__":
    video_path = "/Users/amana1/working_dir/sample_videos/TX_MASTER_FC_Anupamaa_SH4164_S1_E1599_DYN1492441_v2_763507606_900790931_877219001.mp4"
    output_dir = "./output_dir"

    split_video(
        video_path,
        output_dir,
        split_duration=5,
        resize_w=1280,   # Desired width
        resize_h=720     # Desired height
    )



    