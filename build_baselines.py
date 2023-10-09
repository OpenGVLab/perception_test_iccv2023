import os
import json
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import zipfile
import requests


def download_and_unzip(url: str, destination: str):
  """Downloads and unzips a .zip file to a destination.

  Downloads a file from the specified URL, saves it to the destination
  directory, and then extracts its contents.

  If the file is larger than 1GB, it will be downloaded in chunks,
  and the download progress will be displayed.

  Args:
    url (str): The URL of the file to download.
    destination (str): The destination directory to save the file and
      extract its contents.
  """
  if not os.path.exists(destination):
    os.makedirs(destination)

  filename = url.split('/')[-1]
  file_path = os.path.join(destination, filename)

  if os.path.exists(file_path):
    print(f'{filename} already exists. Skipping download.')
    return

  response = requests.get(url, stream=True)
  total_size = int(response.headers.get('content-length', 0))
  gb = 1024*1024*1024

  if total_size / gb > 1:
    print(f'{filename} is larger than 1GB, downloading in chunks')
    chunk_flag = True
    chunk_size = int(total_size/100)
  else:
    chunk_flag = False
    chunk_size = total_size

  with open(file_path, 'wb') as file:
    for chunk_idx, chunk in enumerate(
        response.iter_content(chunk_size=chunk_size)):
      if chunk:
        if chunk_flag:
          print(f"""{chunk_idx}% downloading
          {round((chunk_idx*chunk_size)/gb, 1)}GB
          / {round(total_size/gb, 1)}GB""")
        file.write(chunk)
  print(f"'{filename}' downloaded successfully.")

  with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall(destination)
  print(f"'{filename}' extracted successfully.")

  os.remove(file_path)
     
    
def download():
    data_path = './data/pt'
    model_path = './ckpt'

    train_annot_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/challenge_sound_localisation_train_annotations.zip'
    download_and_unzip(train_annot_url, data_path)
    train_video_feat_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/action_localisation_train_video_features.zip'
    download_and_unzip(train_video_feat_url, data_path)
    train_audio_feat_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/sound_localisation_train_audio_features.zip'
    download_and_unzip(train_audio_feat_url, data_path)

    valid_annot_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/challenge_sound_localisation_valid_annotations.zip'
    download_and_unzip(valid_annot_url, data_path)
    valid_video_feat_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/action_localisation_valid_video_features.zip'
    download_and_unzip(valid_video_feat_url, data_path)
    valid_audio_feat_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/sound_localisation_valid_audio_features.zip'
    download_and_unzip(valid_audio_feat_url, data_path)

    # here we download a pretrained model, this can be commented out and the
    # training command below can be ran instead to train the model from scratch
    model_url = 'https://storage.googleapis.com/dm-perception-test/saved_models/perception_tsl_audio_train_reproduce.zip'
    download_and_unzip(model_url, model_path)