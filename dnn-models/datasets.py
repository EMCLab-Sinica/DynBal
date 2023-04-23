import functools
import sys

import filelock
import numpy as np
import platformdirs
import torch
import torchaudio
import torchvision.datasets
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import TensorDataset

from utils import (
    DataLayout,
    ModelData,
    THIS_DIR,
    download_file,
    extract_archive,
    kws_dnn_model,
)

def load_data_cifar10(train: bool, target_size: tuple[int, int]) -> ModelData:
    xdg_cache_home = platformdirs.user_cache_path()
    transforms = Compose([
        ToTensor(),
        Resize(size=target_size[-2:]),  # H and W from NCHW of ONNX
    ])
    with filelock.FileLock(xdg_cache_home / 'cifar10.lock'):
        dataset = torchvision.datasets.CIFAR10(root=xdg_cache_home, train=train, download=True, transform=transforms)
    return ModelData(dataset=dataset, data_layout=DataLayout.NCHW)

def preprocess_kws_dataset(original_dataset):
    GOOGLE_SPEECH_SAMPLE_RATE = 16000
    # From https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Pretrained_models/labels.txt
    new_labels = '_silence_ _unknown_ yes no up down left right on off stop go'.split(' ')

    import tensorflow as tf

    with open(kws_dnn_model(), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)

    with tf.compat.v1.Session() as sess:
        mfcc_tensor = sess.graph.get_tensor_by_name('Mfcc:0')

        labels = []
        mfccs = None  # The dimensions are unknown yet
        dataset_size = len(original_dataset)
        for n in range(dataset_size):
            # The first few _unknown_ samples are not recognized by Hello Edge's DNN model - use good ones instead
            data = original_dataset[dataset_size - 1 - n]
            waveform, sample_rate, label, _, _ = data
            assert sample_rate == GOOGLE_SPEECH_SAMPLE_RATE
            decoded_wav = np.squeeze(waveform)
            # Some files in speech_commands_v0.02 are less than 1 second. By comparing with corresponding files in
            # speech_commands_test_set_v0.02, zeros are padded at last apparently
            decoded_wav = np.pad(decoded_wav, ((0, GOOGLE_SPEECH_SAMPLE_RATE - len(decoded_wav)),))
            decoded_wav = np.expand_dims(decoded_wav, axis=-1)

            # See the logic at https://github.com/tensorflow/datasets/blob/v4.6.0/tensorflow_datasets/audio/speech_commands.py#L128-L140
            if label in new_labels:
                label = new_labels.index(label)
            elif label in ('_silence_', '_background_noise_'):
                label = new_labels.index('_silence_')
            else:
                label = new_labels.index('_unknown_')
            labels.append(label)

            mfcc = sess.run(mfcc_tensor, {
                'decoded_sample_data:0': decoded_wav,
                'decoded_sample_data:1': GOOGLE_SPEECH_SAMPLE_RATE,
            })
            if mfccs is None:
                mfccs_shape = list(mfcc.shape)
                mfccs_shape[0] = dataset_size
                mfccs = np.zeros(mfccs_shape)
            mfccs[n, :, :] = mfcc

    return mfccs, labels

def load_data_google_speech(train: bool, target_size: tuple[int, int]) -> ModelData:
    xdg_cache_home = platformdirs.user_cache_path()
    with filelock.FileLock(xdg_cache_home / 'SpeechCommands.lock'):
        kws_cache_filename = xdg_cache_home / 'SpeechCommands-cache-v1.pth'
        if kws_cache_filename.exists():
            mfccs, labels = torch.load(kws_cache_filename)
        else:
            original_dataset = torchaudio.datasets.SPEECHCOMMANDS(root=xdg_cache_home, download=True, subset='training' if train else 'testing')
            mfccs, labels = preprocess_kws_dataset(original_dataset)
            torch.save((mfccs, labels), kws_cache_filename)
    dataset = TensorDataset(torch.from_numpy(mfccs.astype(np.float32)), torch.tensor(labels))
    assert dataset[0][0].shape == target_size
    return ModelData(dataset, data_layout=DataLayout.NEUTRAL)

# Inspired by https://blog.csdn.net/bucan804228552/article/details/120143943
def load_har(train: bool, target_size: tuple[int, int]):
    try:
        orig_sys_path = sys.path.copy()
        sys.path.append(str(THIS_DIR / 'deep-learning-HAR' / 'utils'))
        from utilities import read_data, standardize

        archive_dir = download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip',
                                    filename='UCI HAR Dataset.zip', post_processor=functools.partial(extract_archive, subdir='UCI HAR Dataset'))
        X, labels, _ = read_data(archive_dir, split='train' if train else 'test')
        assert X[0].shape == target_size
        _, X = standardize(np.random.rand(*X.shape), X)
        dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(labels-1))
        return ModelData(dataset=dataset, data_layout=DataLayout.NCW)
    finally:
        sys.path = orig_sys_path

