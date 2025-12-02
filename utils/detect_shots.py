import os
import json
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


def process_and_split_shots(video_path, output_dir, scene_list, detect_time, total_start):
    import time
    split_start = time.time()

    # print(f"🎬 Detected {len(scene_list)} scenes in {detect_time:.2f}s:")
    # for i, scene in enumerate(scene_list):
    #     start_time, end_time = scene
    #     print(f"  Scene {i+1}: {start_time.get_timecode()} → {end_time.get_timecode()}")

    if scene_list:
        print(f"\n✂️  Splitting video into '{output_dir}/'...")
        for i, (start_time, end_time) in enumerate(scene_list):
            start_sec = round(start_time.get_seconds(), 3)
            end_sec = round(end_time.get_seconds(), 3)
            duration = end_sec - start_sec
            if duration <= 0.3:
                continue

            output_file = os.path.join(output_dir, f"shot_{start_sec}_{end_sec}.mp4")
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


def detect_and_split_shots(video_path, threshold=30.0):
    """
    Detects scene cuts and splits the video into separate clips.
    threshold: higher = less sensitive, lower = more sensitive
    """

    dir_path = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    json_path = os.path.join(dir_path, video_name, "shots.json")
    
   
    # Initialize video & scene managers
    video_manager = VideoManager([str(video_path)])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Start video
    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)

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

    
    with open(json_path, "w") as f:
        json.dump(scene_json, f, indent=4)
    print(f"\n📝 Scene timestamps saved to: {json_path}")

    shots_dir = os.path.join(dir_path, video_name, "shots")
    os.makedirs(shots_dir, exist_ok=True)

    process_and_split_shots(
        video_path=video_path,
        output_dir=shots_dir,
        scene_list=scene_list,
        detect_time=0.0,
        total_start=0.0
    )


if __name__ == "__main__":
    video_path = '/Users/amana1/working_dir/sample_videos/TX_MASTER_FC_Anupamaa_SH4164_S1_E1599_DYN1492441_v2_763507606_900790931_877219001.mp4'
    threshold = 30.0
    detect_and_split_shots(video_path, threshold)
