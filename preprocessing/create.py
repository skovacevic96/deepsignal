import numpy as np
import time
import os
import librosa
import sys
import tensorflow as tf


def __make_1d_example(signal):
    features_dict = {"data": tf.train.Feature(float_list=tf.train.FloatList(value=signal))}
    features = tf.train.Features(feature=features_dict)
    return tf.train.Example(features=features).SerializeToString()


def create_1d_dataset(root, dataset_path, frame_rate = 16000, step_size = 32768, record_len = 1500):
    """
    Iterate over root folder and acquire all .wav files
    After creating tf records save them in dataset_path
    """
    cnt = 0
    iterator = 0
    tfRecord_filename = os.path.join(dataset_path, f'record_{cnt:04d}.tfrecords')
    writer = tf.io.TFRecordWriter(tfRecord_filename)
    for filename in os.listdir(root):
    if filename.endswith(".wav"):
      y, sr = librosa.load(os.path.join(root, filename), sr=None)
      for i in np.arange(0, len(y), step_size):
        slice_len = min(len(y)-i, step_size)
        signal = np.pad(y[i:(i+step_size)], (0, step_size-slice_len))

        if iterator == record_len:
          iterator = 0
          writer.close()
          sys.stdout.flush()
          cnt += 1

          tfRecord_filename = os.path.join(dataset_path, f'record_{cnt:04d}.tfrecords')
          writer = tf.io.TFRecordWriter(tfRecord_filename)


        iterator += 1
        writer.write(make_1d_example(signal))
