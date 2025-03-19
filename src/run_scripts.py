import os
import time

start_time = time.time()

print("Running preprocess.py...")
os.system("python preprocess.py")

print("Running feature_extraction.py...")
os.system("python feature_extraction.py")

print("Running train_classifier.py...")
os.system("python train_classifier.py")

end_time = time.time()
elapsed_time = int(end_time - start_time)
print(f"All steps completed successfully in {elapsed_time} seconds.")