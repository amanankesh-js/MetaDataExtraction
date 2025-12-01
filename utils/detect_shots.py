import os
import time
import json
from pathlib import Path
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg


def process_and_split_scenes(video_path, output_dir, scene_list, detect_time, total_start):
    import time
    split_start = time.time()

    print(f"🎬 Detected {len(scene_list)} scenes in {detect_time:.2f}s:")
    for i, scene in enumerate(scene_list):
        start_time, end_time = scene
        print(f"  Scene {i+1}: {start_time.get_timecode()} → {end_time.get_timecode()}")

    if scene_list:
        print(f"\n✂️  Splitting video into '{output_dir}/'...")
        for i, (start_time, end_time) in enumerate(scene_list):
            start_sec = round(start_time.get_seconds(), 3)
            end_sec = round(end_time.get_seconds(), 3)
            duration = end_sec - start_sec
            if duration <= 0.3:
                continue

            output_file = output_dir / f"shot_{start_sec}_{end_sec}.mp4"
            cmd = (
                f"ffmpeg -hide_banner -loglevel error "
                f"-ss {start_sec} -to {end_sec} -i \"{video_path}\" "
                f"-c copy \"{output_file}\""
            )
            os.system(cmd)
            print(f"✅ Saved: {output_file.name} ({duration:.2f}s)")

        split_time = time.time() - split_start
        print(f"\n✅ All scenes saved in: {output_dir.resolve()}")
    else:
        print("⚠️ No scenes detected. Try lowering the threshold (e.g. 25.0 or 20.0).")
        split_time = 0.0

    total_time = time.time() - total_start
    print("\n⏱️  Timing Summary")
    print(f"   🔹 Detection time: {detect_time:.2f}s")
    print(f"   🔹 Splitting time: {split_time:.2f}s")
    print(f"   🔹 Total time: {total_time:.2f}s")


def detect_and_split_scenes(video_path, threshold=30.0, method_name="pyscenedetect"):
    """
    Detects scene cuts and splits the video into separate clips.
    threshold: higher = less sensitive, lower = more sensitive
    """

    # Extract video name and prepare output directory
    from datetime import datetime

    video_stem = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = Path("output") / video_stem / timestamp
    shots_dir = base_dir / "shots"
    json_dir = base_dir / "json"

    shots_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)


    print(f"📹 Processing: {video_path}")
    print(f"💾 Saving scenes to: {base_dir}\n")

    total_start = time.time()

    # Initialize video & scene managers
    video_manager = VideoManager([str(video_path)])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Start video
    video_manager.start()

    # --- Scene Detection ---
    detect_start = time.time()
    scene_manager.detect_scenes(frame_source=video_manager)
    detect_time = time.time() - detect_start

    # Get list of scene timecodes
    scene_list = scene_manager.get_scene_list()
    scene_json = []
    for i, scene in enumerate(scene_list):
        start_time, end_time = scene
        start_sec = round(start_time.get_seconds(), 3)
        end_sec = round(end_time.get_seconds(), 3)
        scene_json.append({
            "scene_index": i + 1,
            "start_timecode": start_time.get_timecode(),
            "end_timecode": end_time.get_timecode(),
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "duration": round(end_sec - start_sec, 3)
        })

    json_path = json_dir / f"{video_stem}_scenes.json"
    with open(json_path, "w") as f:
        json.dump(scene_json, f, indent=4)
    print(f"\n📝 Scene timestamps saved to: {json_path}")

    #process_and_split_scenes(video_path, shots_dir, scene_list, detect_time, total_start)


if __name__ == "__main__":
    video_path = '/Users/amana1/working_dir/sample_videos/TX_MASTER_FC_Anupamaa_SH4164_S1_E1599_DYN1492441_v2_763507606_900790931_877219001.mp4'
    threshold = 30.0
    detect_and_split_scenes(video_path, threshold)
