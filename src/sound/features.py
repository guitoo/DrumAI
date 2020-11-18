from librosa.core import resample, load
from librosa.util import fix_length
from librosa import feature
import numpy as np
import timbral_models
import tensorflow_hub as hub

SAMPLE_RATE = 16000

vggish_model = hub.load('https://tfhub.dev/google/vggish/1')
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def load_sample(path):
    array, _ = load(path, sr=SAMPLE_RATE)
    return array

def vggish_embedding(path=None, array=None):
    if array is None:
        array = load_sample(path)
    array = fix_length(array, SAMPLE_RATE)
    return vggish_model(array).numpy()[0]

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
    try:
        result = timbral_models.timbral_hardness(path)
    except:
        return -1
    if np.isnan(result):
        return -1
    return result

def depth(path):
    try:
        result = timbral_models.timbral_depth(path, clip_output=True)
    except:
        return -1
    if np.isnan(result):
        return -1
    return result

def brightness(path):
    try:
        result = timbral_models.timbral_brightness(path, clip_output=True)
    except:
        return -1
    if np.isnan(result):
        return -1
    return result

def roughness(path):
    try:
        result = timbral_models.timbral_roughness(path)
    except:
        return -1
    if np.isnan(result):
        return -1
    return result

def warmth(path):
    try:
        result = timbral_models.timbral_warmth(path)
    except:
        return -1
    if np.isnan(result):
        return -1
    return result


def sharpness(path):
    try:
        result = timbral_models.timbral_sharpness(path)
    except:
        return -1
    if np.isnan(result):
        return -1
    return result


def boominess(path):
    try:
        result = timbral_models.timbral_booming(path)
    except:
        return -1
    if np.isnan(result):
        return -1
    return result




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