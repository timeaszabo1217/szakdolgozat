import os

print("Running preprocess.py...")
os.system("python scripts/preprocess.py")

print("Running feature_extraction.py...")
os.system("python scripts/feature_extraction.py")

print("Running train_classifier.py...")
os.system("python scripts/train_classifier.py")

print("All steps completed successfully.")
