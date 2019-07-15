import librosa
import numpy as np
from mp3_tagger import MP3File, VERSION_1, VERSION_2, VERSION_BOTH
import json
import matplotlib.pyplot as plt
import librosa.display

import warnings

def _load(fn):
  x, sr = librosa.load(fn, offset=60, duration=20.0)
  return {"fn": fn, "onset": librosa.onset.onset_strength(x, sr=sr), "sr": sr, "x":x}

def load(fn):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return _load(fn)

def _bpm(h):
  return librosa.beat.tempo(onset_envelope=h["onset"], sr=h["sr"])

def bpm(fn):
  return _bpm(load(fn))

def plot_tempograph(fn):
  h=load(fn)
  dtempo = librosa.beat.tempo(onset_envelope=h["onset"], sr=h["sr"],
      aggregate=None)
  hop_length = 512
  plt.figure()
  tg = librosa.feature.tempogram(onset_envelope=h["onset"], sr=h["sr"],
      hop_length=hop_length)
  librosa.display.specshow(tg, x_axis='time', y_axis='tempo')
  plt.plot(librosa.frames_to_time(np.arange(len(dtempo))), dtempo,
      color='w', linewidth=1.5, label='Tempo estimate')
  plt.title('Dynamic tempo estimation')
  plt.legend(frameon=True, framealpha=0.75)


def plot_tempograph_fast(h):
  dtempo = librosa.beat.tempo(onset_envelope=h["onset"], sr=h["sr"],
      aggregate=None)
  hop_length = 512
  plt.figure()
  tg = librosa.feature.tempogram(onset_envelope=h["onset"], sr=h["sr"],
      hop_length=hop_length)
  plt.plot(librosa.frames_to_time(np.arange(len(dtempo))), dtempo,
      color='b', linewidth=1.5, label='Tempo estimate')
  plt.title('Dynamic tempo estimation')
  plt.legend(frameon=True, framealpha=0.75)

def to_h(x):
  sr = 22050
  return {"onset": librosa.onset.onset_strength(x, sr=sr), "sr": sr, "x":x}

def quickview(h):
  #h = load(fn)
  print(_bpm(h))
  plot_tempograph_fast(h)

from os import listdir
from os.path import isfile, join
import random
import re

def all_mp3_fns():
  return [join("data/songs", f) for f in listdir("data/songs")]

def imagenet_results():
  return json.loads(open("imagenet_results.json", "rb").read())

import os

def only_lindyhop():
  r = os.popen('''mid3v2 -l data/songs/*.mp3 | grep -B 2 lindy_hop | grep -v TALB | grep -v 'TCON'  | grep -v -- "--" |  perl -pe "s,^IDv2 tag info for data/songs/,./,"''').read()
  h = {}
  a = [s.strip() for s in r.splitlines()]
  for e in a: h[e] = True
  mp3s = all_mp3_fns()
  indexes = []
  for i, fn in enumerate(mp3s):
    s = re.sub(r'^data/songs/', "./", fn)
    if s in h:
      indexes.append(i)
  ih = {}
  for e in indexes: ih[e] = True
  return {
    'filenames': h,
    'indexes': indexes,
    'indexesh': ih
  }

def all_mp3_tags():
  r = []
  for fn in all_mp3_fns():
    mp3 = MP3File(fn)
    r.append(mp3)
    break
  return r

def load_all_mp3s():
  c = 0
  bpms = []
  data = []
  fns = all_mp3_fns()
  for fn in fns:
    h = load(fn)
    data.append(h["x"])
    m = re.search('(\d+)', fn)
    bpms.append(m.group(1))
    c += 1
    if c % 50 == 0:
      print(c)
  return {"bpms": bpms, "data": data, 'filenames': fns}
  

def random_mp3_fn():
  return random.choice([fn for fn in all_mp3_fns() if isfile(fn)])

def load_random_mp3():
  return load(random_mp3_fn())


def spectrogram(audio):
  S = librosa.stft(audio)
  M = librosa.core.magphase(S)[0]
  return librosa.amplitude_to_db(M, ref=np.max)

def plot_spectrogram(spect):
  compspect = spect
  (h, w) = compspect.shape
  fig = plt.figure(figsize=(w/100, h/100))
  ax = plt.subplot(111)
  ax.set_frame_on(False)
  plt.axis('off')
  ax.axes.get_xaxis().set_visible(False)
  ax.axes.get_yaxis().set_visible(False)
  librosa.display.specshow(compspect, y_axis='linear')
  return fig

def save_spectrogram(spect, fn):
  png_fn = "data/spectrographs/{}".format(fn.replace(".mp3", ".png"))
  if isfile(png_fn): return
  print(png_fn)
  fig = plot_spectrogram(spect)
  fig.savefig(png_fn, dpi=100, bbox_inches='tight', pad_inches=0.0)
  plt.close(fig)

def all_spectrograph_filenames(d):
  return ["data/spectrographs/{}.png".format(
      os.path.splitext(os.path.basename(fn))[0]) for fn in d['filenames']]

def split_spectrographs__create_symlinks(where, fns):
  for fn in fns:
    c = os.getcwd()
    src  =  "{}/{}".format(c, fn)
    b = os.path.basename(fn)
    m = re.search('(\d+)', b)
    bpm = m.group(1)
    dest_dir = "{}/data/{}/{}".format(c, where, bpm, b)
    os.makedirs(dest_dir, exist_ok=True)
    dest = "{}/{}".format(dest_dir, b)
    #print("{} -> {}".format(src, dest))
    os.symlink(src, dest)

def split_spectrographs_into_train_and_validation_sets(d):
  os.system("rm -rf data/{train,valid}")
  os.makedirs("data/train", exist_ok=True)
  os.makedirs("data/valid", exist_ok=True)
  fns = np.array(all_spectrograph_filenames(d))
  trn_keep = np.random.rand(len(fns)) > 0.4
  train = fns[trn_keep]
  val = fns[~trn_keep]
  split_spectrographs__create_symlinks("train", train)
  split_spectrographs__create_symlinks("valid", val)

