from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth() # Opens a browser to log in
drive = GoogleDrive(gauth)

file_to_upload = "output/resnext101_cascade_60k/model_final.pth"
folder_id = "YOUR_DRIVE_FOLDER_ID" # Get this from the Drive URL

file_drive = drive.CreateFile({'title': 'resnext_model.pth', 'parents': [{'id': folder_id}]})
file_drive.SetContentFile(file_to_upload)
file_drive.Upload()
print("Upload Complete!")