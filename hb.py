import librosa
import networkx as nx
import numpy as np
import os
from pyvis.network import Network
import re
import sys
import warnings
import webbrowser

# ----- SETTINGS ----- #

valid_file_types = [".wav", ".flac"]
out_filename = 'hit-browser.html'
n_matches = 2
sim_thresh = 0.98
n_mfcc_bins = 32

# Weights of each characteristic being compared.
coef = {
  "duration": 1,
  "attack":   3,
  "decay":    2,
  "sustain":  1,
  "release":  0,
  "pitch":    2,
  "mfcc":     1
}

# ----- HELPER FUNCTIONS ----- #

# Linearly normalize any data to between 0 and 1
def normalize_linear(arr):
  return ((arr - np.min(arr)) / np.max(arr))

# Print an error message and/or how to use this program.
def print_usage(msg):
  if (msg): print("ERROR: " + msg)
  print("USAGE: python3 hb.py <path to hit library>")
  quit()

# ----- READ ARGUMENTS ----- #

# Check the number of aruments provided.
if (len(sys.argv) != 2):
  print_usage("Invalid number of arguments.")

# Check that the path is valid.
if os.path.isdir(sys.argv[1]):
  hit_lib = os.path.abspath(sys.argv[1])
else:
  print_usage("Not a valid path.")

# ----- LOAD FILE DATA -----#

# Inititialize the array that will store the data about each hit file.
hit_data = []

# Find the path to all the valid audio files.
for root, dirs, files in os.walk(hit_lib, topdown = True, followlinks = True):
  for file in files:
    # Get the file's absolute path.
    file_path = root + "/" + file

    # Get the file's extension and ignore the files without a valid extension.
    file_ext = re.findall("\.[a-z,0-9]+$", file)
    if len(file_ext) == 0 or file_ext[0] not in valid_file_types:
      continue

    warnings.filterwarnings("ignore") # Turn off warnings.

    # Load the file data and trim the leading and trailing silence.
    try:
      file_data_orig, file_sr = librosa.load(file_path, sr = None)
      file_data = librosa.effects.trim(file_data_orig, top_db = 5)[0]
    except:
      print("WARNING: Failed to load: " + file_path)
      continue

    # Get the file's duration.
    file_dur = librosa.get_duration(y = file_data, sr = file_sr)
    if (file_dur <= 0):
      continue

    # Calculate the mfcc of the file.
    file_mfcc = np.array(librosa.feature.mfcc(\
      y = file_data,\
      sr = file_sr,\
      n_mfcc = n_mfcc_bins\
    ))
    file_mfcc_bin = np.argmax(np.mean(file_mfcc, axis = 1))

    warnings.resetwarnings() # Turn warnings back on.

    # Calculate the STFT of the file and the total power over time.
    file_stft = np.array(librosa.stft(y = file_data, n_fft = 255, hop_length = 1))

    # Determine the average pitch of the file.
    file_pitch_bin = np.argmax(np.mean(file_stft, axis = 1))

    # Calculate the attack. (Time between 0% and 100% power)
    pow_over_time = np.sum(file_stft, axis = 0)
    file_max_pow = np.max(pow_over_time)
    file_attack = np.argmax(pow_over_time)

    # Calculate the sustain. (The level at which most of the time is spent)
    file_sustain_level = np.mean(pow_over_time[file_attack:])
    sustain_margin = file_max_pow * 0.2

    # Calculate the decay. (Time between 100% power and sustain level)
    file_decay = np.argmax(pow_over_time[file_attack:] < file_sustain_level + sustain_margin)

    # Calculate the release. (Amount of time between the sustain level and 0%)
    file_sustain = np.argmax(pow_over_time[file_attack + file_decay:] < file_sustain_level - sustain_margin)
    file_release = len(pow_over_time[file_attack + file_decay + file_sustain:])

    # Add the file to the list of files.
    hit_data.append({
      "path": file_path[len(hit_lib):],
      "dur": file_dur,
      "attack": file_attack / file_sr,
      "decay": file_decay / file_sr,
      "sustain": file_sustain / file_sr,
      "release": file_release / file_sr,
      "pitch": file_pitch_bin,
      "mfcc": file_mfcc_bin
    })

# Make sure that there were some hit loaded.
if (len(hit_data) == 0):
  print_usage("No hits found.")

# ----- CALCULATE EDGES ----- #

# Compare durations.
hit_dur = np.array([hit['dur'] for hit in hit_data])
hit_dur_dist = abs(hit_dur - hit_dur.reshape(-1, 1))

# Compare attack.
hit_attack = np.array([file['attack'] for file in hit_data])
attack_dist = abs(hit_attack - hit_attack.reshape(-1, 1))

# Compare decay.
hit_decay = np.array([file['decay'] for file in hit_data])
decay_dist = abs(hit_decay - hit_decay.reshape(-1, 1))

# Compare sustain.
hit_sustain = np.array([file['sustain'] for file in hit_data])
sustain_dist = abs(hit_sustain - hit_sustain.reshape(-1, 1))

# Compare release.
hit_release = np.array([file['release'] for file in hit_data])
release_dist = abs(hit_release - hit_release.reshape(-1, 1))

# Compare pitch.
hit_pitch = np.array([file['pitch'] for file in hit_data])
pitch_dist = abs(hit_pitch - hit_pitch.reshape(-1, 1))

# Compare mfcc.
hit_mfcc = np.array([file['mfcc'] for file in hit_data])
mfcc_dist = abs(hit_mfcc - hit_mfcc.reshape(-1, 1))

# Calculate the similarity of the two beats.
total_dist = \
  (coef["duration"] * normalize_linear(hit_dur_dist)) + \
  (coef["attack"] * normalize_linear(attack_dist)) + \
  (coef["decay"] * normalize_linear(decay_dist)) + \
  (coef["sustain"] * normalize_linear(sustain_dist)) + \
  (coef["release"] * normalize_linear(release_dist)) + \
  (coef["pitch"] * normalize_linear(pitch_dist)) + \
  (coef["mfcc"] * normalize_linear(mfcc_dist))

similarity = 1 - normalize_linear(total_dist)
np.fill_diagonal(similarity, 0)

# ----- CREATE NETWORKX NETWORK ----- #

# Set up the graph.
hit_graph = nx.Graph()

# Add the nodes to the graph.
for (i, hit_file) in enumerate(hit_data):
  hit_graph.add_node(i, path = hit_file['path'], size = 40)

# Connect each hit with its n most similar hits.
for i, p in enumerate([np.argsort(s) for s in similarity]):
  for j in range(n_matches):
    sim = similarity[i][int(p[-(j + 1)])]
    if sim < sim_thresh: continue
    hit_graph.add_edge(i, int(p[-(j + 1)]), value=sim)

for (i, n) in enumerate(nx.spring_layout(hit_graph)):
  print(n[0])
  # print(hit_graph.nodes[i])
  # print(hit_graph.nodes[i]["path"])
  # hit_graph.nodes[i]["x"] = n[i]


# Print out the number of nodes and edges.
print(hit_graph)

# ----- CREATE PYVIZ NETWORK ----- #

# Initialize the graph.
net = Network()

# Set the options for the graph.
net.set_template('./template.html')
net.set_options("""
var options = {
  "autoResize": true,
  "locale": "en",
  "edges": {
    "chosen": false
  },
  "layout": {
    "improvedLayout": false
  },
  "interaction": {
    "dragNodes": false
  },
  "physics": {
    "enabled": false,
    "solver": "barnesHut",
    "timestep": 0.5,
    "repulsion": {
      "centralGravity": 0.00,
      "springLength": 200,
      "springConstant": 0.04,
      "nodeDistance": 100,
      "damping": 0.09
    },
    "barnesHut": {
      "theta": 0.5,
      "gravitationalConstant": -3000,
      "centralGravity": 0.0,
      "springLength": 150,
      "springConstant": 0.05,
      "damping": 0.09,
      "avoidOverlap": 0
    }
  }
}
""")
net.heading = """
{
  "root_path": "%s"
}
""" % (hit_lib)

# Import the nodes from the NetworkX graph.
net.from_nx(hit_graph)

# Write the graph to an HTML file and open it.
# net.show(out_filename)
# webbrowser.open('file:///' + os.getcwd() + '/' + out_filename)
