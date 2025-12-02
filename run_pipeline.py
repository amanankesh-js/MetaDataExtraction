from multiprocessing import Process


def run_file(f):
    import subprocess
    subprocess.run(["python", f])

if __name__ == "__main__":
    n_procs_per_stage = 2
    files = ["download_stage.py", "character_detection_stage.py", "inference_stage.py", "shot_description_stage.py"] * n_procs_per_stage
    processes = []
    for f in files:
        p = Process(target=run_file, args=(f,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # wait for all to finish

