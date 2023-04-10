import tkinter as tk
import librosa
import os # for reading directories
import re # for regex

hit_lib = "/Library/Audio/Sounds/Drum Kits"

valid_file_types = [".wav", ".flac"]
resample_rate = 22050

# for root, dirs, files in os.walk(path):
#   print(dirs)

def load_files(path):
  arr = []

  path_walk = os.walk(path, topdown=True, followlinks=True)
  for root, dirs, files in path_walk:
    for file in files:
      # Get the absolute path.
      file_path = root + "/" + file

      # Ignore the files that dont have a valid extension.
      file_ext = re.findall("\.[a-z,0-9]+$", file_path)
      if len(file_ext) == 0 or file_ext[0] not in valid_file_types:
        continue

      # Load the file data.
      try:
        file_data, file_sr = librosa.load(file_path, sr=None)
      except:
        print("Failed to load: " + file_path)
        continue

      # Add the file to the list of files.
      arr.append({
        "name": file,
        "path": file_path,
        "ext": file_ext[0],
        "sr": file_sr,
        "data": file_data
      })

      # print()

  return arr

hit_files = load_files(hit_lib)

print(hit_files)
  
# window = tk.Tk()
# label = tk.Label(text="Python rocks!")
# label.pack()

# window.mainloop()