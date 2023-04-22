import networkx as nx
import numpy as np
import librosa
import os
import re
import webbrowser
from pyvis.network import Network
import warnings
from scipy import stats
import sys
from pathlib import Path

# ----- SETTINGS ----- #

valid_file_types = [".wav", ".flac"]
out_filename = 'hit-browser.html'
n_matches = 3
sim_thresh = 0.98

coef_pitch = 2
n_pitch_bins = 64

coef_timbre = 4
n_timbre_bins = 32

coef_duration = 2

coef_attack = 4

# ----- HELPER FUNCTIONS ----- #

# Linearly normalize any data to between 0 and 1
def normalize_linear(arr):
  return ((arr - np.min(arr)) / np.max(arr))

def print_usage(msg):
  if (msg): print("ERROR: " + msg)
  print("USAGE: python3 main.py <path to hit library>")
  quit()

# ----- READ ARGUMENTS ----- #

# Check the number of aruments.
if (len(sys.argv) != 2): 
  print_usage("Invalid number of arguments.")

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

    # Get the file's extension and ignore the files that dont have a valid extension.
    file_ext = re.findall("\.[a-z,0-9]+$", file)
    if len(file_ext) == 0 or file_ext[0] not in valid_file_types:
      continue

    # Turn off warnings.
    warnings.filterwarnings("ignore") 

    # Load the file data and trim the leading and trailing silence.
    try:
      file_data_orig, file_sr = librosa.load(file_path, sr = None)
      file_data, file_data_trim_index = librosa.effects.trim(file_data_orig, top_db = 5)
    except:
      print("Failed to load: " + file_path)
      continue

    # Get the file's duration.
    file_dur = librosa.get_duration(y = file_data, sr = file_sr)
    if (file_dur <= 0): continue

    # Get the Attack (time between 0% and 100% power)
    file_attack_i = np.argmax(np.abs(file_data))
    file_attack = file_attack_i / file_sr

    # Get the Sustain (The level at which most of the time is spent)
    # file_sustain = stats.mode(file_data)

    # Get the decay (Time between 100% power and sustain level)

    # Get teh release (Amount of time between the sustain level and 0%)

    # Calculate the mfcc of the file.
    file_mfcc = np.array(librosa.feature.mfcc(y = file_data, sr = file_sr, n_mfcc = n_timbre_bins))
    file_timbre_bin = np.argmax(np.mean(file_mfcc, axis = 1))

    # Calculate the stft of the file.
    file_stft = np.array(librosa.stft(y = file_data, n_fft = 2 * (n_pitch_bins - 1)))
    file_pitch_bin = np.argmax(np.mean(file_stft, axis = 1))

    warnings.resetwarnings() # Turn warnings back on.

    # Add the file to the list of files.
    hit_data.append({
      "name": file,
      "path": file_path[len(hit_lib):],
      "dur": file_dur,
      "attack": file_attack,
      "pitch": file_pitch_bin,
      "timbre": file_timbre_bin
    })

if (len(hit_data) == 0):
  print_usage("No hits found.")

# ----- CALCULATE EDGES ----- #

# Compare durations.
hit_dur = np.array([hit['dur'] for hit in hit_data])
hit_dur_dist = normalize_linear(abs(hit_dur - hit_dur.reshape(-1, 1)))

# Compare attack.
hit_attack = np.array([file['attack'] for file in hit_data])
attack_dist = normalize_linear(abs(hit_attack - hit_attack.reshape(-1, 1)))

# Compare pitch.
hit_pitch = np.array([file['pitch'] for file in hit_data])
pitch_dist = normalize_linear(abs(hit_pitch - hit_pitch.reshape(-1, 1)))

# Compare timbre.
hit_timbre = np.array([file['timbre'] for file in hit_data])
timbre_dist = normalize_linear(abs(hit_timbre - hit_timbre.reshape(-1, 1)))

# Calculate the similarity of the two beats.
total_dist = \
  (coef_duration * hit_dur_dist) + \
  (coef_attack * attack_dist) + \
  (coef_pitch * pitch_dist) + \
  (coef_timbre * timbre_dist)
similarity = 1 - normalize_linear(total_dist)
np.fill_diagonal(similarity, 0)

# ----- CREATE NETWORKX NETWORK ----- #

# Set up the graph.
hit_graph = nx.Graph()

# Add the nodes to the graph.
for (i, hit_file) in enumerate(hit_data):
  hit_graph.add_node(
    i,
    label = None,
    path = hit_file['path'],
    size = 40
  )

# Connect each hit with its n most similar hits.
for i, p in enumerate([np.argsort(s) for s in similarity]):
  for j in range(1, n_matches + 1):
    sim = similarity[i][int(p[-j])]
    if sim < sim_thresh: continue
    # if (i < int(p[-j])): hit_graph.add_edge(i, int(p[-j]))
    hit_graph.add_edge(i, int(p[-j]), value=sim)

print(hit_graph)

# ----- CREATE PYVIZ NETWORK ----- #

net = Network()

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

net.from_nx(hit_graph)

net.show(out_filename)
webbrowser.open('file:///' + os.getcwd() + '/' + out_filename)
