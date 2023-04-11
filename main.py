import networkx as nx
import librosa
import os # for reading directories
import re # for regex
import webbrowser
from pyvis.network import Network



# ----- Settings ----- #
hit_lib = "/Library/Audio/Sounds/Drum Kits"
valid_file_types = [".wav", ".flac"]
resample_rate = 22050

# ----- Load Files -----#
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

  return arr



# ----- Main ----- #
hit_files = load_files(hit_lib)
# print(hit_files)

# Set up the graph.
G = nx.Graph()

# Add the nodes to the graph.
for f in hit_files:
  G.add_node(f["name"], path=f['path'], ext=f['ext'])

# Add the edges to the graph.



out_filename = 'net.html'

net = Network()
net.set_template('./template.html')
# net.toggle_physics(False)
net.from_nx(G)
net.show(out_filename)
webbrowser.open_new_tab('file:///' + os.getcwd() + '/' + out_filename)
