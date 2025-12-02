from multiprocessing import Process
import subprocess

def run_multiple_python(files):
    """
    Run a list of Python files in parallel using multiprocessing.

    :param files: list of Python script filenames
    :param n_procs_per_stage: how many times each file should run in parallel
    """
    processes = []

    for f in files:
        p = Process(target=subprocess.run, args=(["python", f],))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # wait for all files to finish

if __name__ == "__main__":
    n_procs_per_stage = 3
    files = [
        "download_stage.py",
        "character_detection_stage.py",
        "inference_stage.py",
        "shot_description_stage.py"
    ]
    run_multiple_python(["download_stage.py"]* n_procs_per_stage )
    run_multiple_python(["character_detection_stage.py"] * n_procs_per_stage)
    run_multiple_python(["inference_stage.py"] * n_procs_per_stage)
    run_multiple_python(["shot_description_stage.py"] * n_procs_per_stage)
