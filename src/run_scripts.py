import subprocess
import time

RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

start_time = time.time()

scripts = ["preprocess.py", "feature_extraction.py", "train_classifier.py", "test_classifier.py"]

for script in scripts:
    print(f"{BLUE}➜ Running {script}...{RESET}", flush=True)
    time.sleep(3)

    try:
        process = subprocess.Popen(
            ["python", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            executable="C:\\Users\\timea\\OneDrive\\Documents\\GitHub\\Szakdolgozat\\venv\\Scripts\\python.exe"
        )

        for line in process.stdout:
            print(line, end='', flush=True)

        for line in process.stderr:
            print(f"{RED}{line}{RESET}", end='', flush=True)

        process.wait()

        if process.returncode == 0:
            print(f"{GREEN}✔ Done{RESET}\n", flush=True)
            time.sleep(1)
        else:
            print(f"{RED}✘ Error: {script} failed with exit code {process.returncode}{RESET}", flush=True)
            print(f"{RED}Stopping execution.{RESET}\n", flush=True)
            break

    except Exception as e:
        print(f"\n{RED}✘ Error: {script} encountered an exception: {str(e)}{RESET}", flush=True)
        print(f"{RED}Stopping execution.{RESET}\n", flush=True)
        break

end_time = time.time()
elapsed_time = int(end_time - start_time)
print(f"{GREEN}✔ All steps completed successfully in {elapsed_time} seconds.{RESET}", flush=True)
