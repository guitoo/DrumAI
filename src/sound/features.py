from librosa.core import resample, load
from librosa.util import fix_length
from librosa import feature
import numpy as np
import timbral_models
import tensorflow_hub as hub
import os

SAMPLE_RATE = 16000

vggish_model = hub.load('https://tfhub.dev/google/vggish/1')
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def load_sample(path, sr=SAMPLE_RATE):
    array, _ = load(path, sr=sr)
    return array

def vggish_embedding(path=None, array=None):
    if array is None:
        array = load_sample(path)
    array = fix_length(array, SAMPLE_RATE)
    return vggish_model(array).numpy().reshape(-1)

def yamnet_embedding(path=None, array=None):
    if array is None:
        array = load_sample(path)
    array = fix_length(array, SAMPLE_RATE)
    _, yam_emb, _ = yamnet_model(array)
    return yam_emb.numpy()[0]

def mfcc(path=None, array=None, hop_length=160, n_mfcc=64, n_fft=512, win_length=400):
    if array is None:
        array = load_sample(path)
    array = fix_length(array, SAMPLE_RATE)
    mfcc = feature.mfcc(array, sr=SAMPLE_RATE, hop_length=hop_length, n_mfcc=n_mfcc, n_fft=n_fft, win_length=win_length)
    mfcc = np.ascontiguousarray(mfcc[:,:96])
    return mfcc

# def timbre(path):
#     timbre = timbral_extractor(path, verbose=False)
#     timbre.pop('reverb')
#     return timbre

def hardness(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    try:
        result = timbral_models.timbral_hardness(path)
    except:
        return -1
    if np.isnan(result):
        return -1
    return result

def depth(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    try:
        result = timbral_models.timbral_depth(path, clip_output=True)
    except:
        return -1
    if np.isnan(result):
        return -1
    return result

def brightness(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    try:
        result = timbral_models.timbral_brightness(path, clip_output=True)
    except:
        return -1
    if np.isnan(result):
        return -1
    return result

def roughness(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    try:
        result = timbral_models.timbral_roughness(path)
    except:
        return -1
    if np.isnan(result):
        return -1
    return result

def warmth(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    try:
        result = timbral_models.timbral_warmth(path)
    except:
        return -1
    if np.isnan(result):
        return -1
    return result


def sharpness(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    try:
        result = timbral_models.timbral_sharpness(path)
    except:
        return -1
    if np.isnan(result):
        return -1
    return result


def boominess(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    try:
        result = timbral_models.timbral_booming(path)
    except:
        return -1
    if np.isnan(result):
        return -1
    return result

def contrast(path=None, array=None, sr=SAMPLE_RATE):
    if array is None:
        array = load_sample(path, sr)
    array = fix_length(array, sr)
    return feature.spectral_contrast(array, sr=sr, fmin=62, n_bands=7)

def zero_crossing_rate(path=None, array=None, sr=SAMPLE_RATE):
    if array is None:
        array = load_sample(path, sr)
    array = fix_length(array, sr)
    return feature.zero_crossing_rate(array).reshape(-1)

def spectral_flatness(path=None, array=None, sr=SAMPLE_RATE):
    if array is None:
        array = load_sample(path, sr)
    array = fix_length(array, sr)
    return feature.spectral_flatness(array).reshape(-1)


def fingerprint(path=None, array=None, **kwargs):
    if array is None:
        array = load_sample(path)
    fprint = mfcc(array=array, hop_length=320, n_mfcc=32, **kwargs)
    fprint = np.ascontiguousarray(fprint[:,:48])
    return fprint

def fingerprint_to_sound(array):
    sound = feature.inverse.mfcc_to_audio(array, hop_length=320, n_mels=64, sr=SAMPLE_RATE)#.astype(np.float32)
    # print(sound.dtype)
    return sound