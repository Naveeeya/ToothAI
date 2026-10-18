import wandb
import os

# 1. Start the session
run = wandb.init(project="Tooth-ai", job_type="model-upload")

# 2. Create the Artifact
artifact = wandb.Artifact(name="opg-model-weights", type="model")

# 3. List of all folders from your screenshot
model_folders = [
    "cascade_r101_70k",
    "resnet50_9class_20k",
    "resnext101_9class_40k",
    "resnext101_cascade_60k",
    "rtx4060_48k"
]

# 4. Loop through and add if the file exists
for folder in model_folders:
    file_path = f"output/{folder}/model_final.pth"
    if os.path.exists(file_path):
        # We name them based on the folder to avoid the "same path twice" error
        artifact.add_file(file_path, name=f"{folder}.pth")
        print(f"Added: {folder}.pth")
    else:
        
        print(f"Skipped: {file_path} not found.")

# 5. Upload and Finish
print("Starting upload to W&B...")
run.log_artifact(artifact)
run.finish()