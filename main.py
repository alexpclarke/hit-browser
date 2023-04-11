import networkx as nx
import librosa
import os # for reading directories
import re # for regex
import webbrowser
from pyvis.network import Network
import warnings

# ----- Settings ----- #
hit_lib = "/Library/Audio/Sounds/Drum Kits"
valid_file_types = [".wav", ".flac"]
out_filename = 'net.html'

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
      # warnings.filterwarnings("error")
      warnings.filterwarnings("ignore")
      try:
        file_data, file_sr = librosa.load(file_path, sr=None)
        file_data_trim, file_data_trim_index = librosa.effects.trim(file_data)
      except:
        print("Failed to load: " + file_path)
        continue
      warnings.resetwarnings()

      # Add the file to the list of files.
      arr.append({
        "name": file,
        "path": file_path,
        "ext": file_ext[0],
        "sr": file_sr,
        "dur": librosa.get_duration(y = file_data_trim, sr = file_sr),
        "data": file_data_trim
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
for i in range(0, len(hit_files)):
  for j in range(i + 1, len(hit_files)):
    # Check the percent difference in length.
    dur1 = hit_files[i]['dur']
    dur2 = hit_files[j]['dur']
    perc_diff = abs(dur1 - dur2) / ((dur1 + dur2) / 2)
    if (perc_diff < 0.005):
      G.add_edge(hit_files[i]['name'], hit_files[j]['name'])


net = Network(cdn_resources='remote')
net.set_template('./template.html')
# net.toggle_physics(False)
net.from_nx(G)
net.show(out_filename)
webbrowser.open('file:///' + os.getcwd() + '/' + out_filename)
