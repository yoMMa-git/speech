# Exercises for laboratory work


# Import of modules
import numpy as np
from scipy.fftpack import dct


def split_meta_line(line, delimiter=' '):
    parts = line.strip().split(delimiter)
    speaker_id = parts[0]
    gender = parts[1]
    file_path = parts[2]
    return speaker_id, gender, file_path


def preemphasis(signal, pre_emphasis=0.97):
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return emphasized_signal


def framing(emphasized_signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01):
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(emphasized_signal)

    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    frames = pad_signal[indices.astype(np.int32)]

    # Hamming window
    frames *= np.hamming(frame_length)

    return frames


def power_spectrum(frames, NFFT=512):
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)
    return pow_frames


def compute_fbank_filters(nfilt=40, sample_rate=16000, NFFT=512):
    low_freq_mel = 0
    high_freq = sample_rate / 2

    # Hz → Mel
    high_freq_mel = 2595 * np.log10(1 + high_freq / 700)

    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)

    # Mel → Hz
    hz_points = 700 * (10**(mel_points / 2595) - 1)

    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))

    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return fbank


def compute_fbanks_features(pow_frames, fbank):
    filter_banks_features = np.dot(pow_frames, fbank.T)

    filter_banks_features = np.where(filter_banks_features == 0,
                                     np.finfo(float).eps,
                                     filter_banks_features)

    filter_banks_features = np.log(filter_banks_features)

    return filter_banks_features


def compute_mfcc(filter_banks_features, num_ceps=20):
    mfcc = dct(filter_banks_features, type=2, axis=1, norm='ortho')[:, :num_ceps]
    return mfcc


def mvn_floating(features, LC, RC, unbiased=False):
    nframes, dim = features.shape

    LC = min(LC, nframes - 1)
    RC = min(RC, nframes - 1)

    n = (np.r_[np.arange(RC + 1, nframes), np.ones(RC + 1) * nframes] -
         np.r_[np.zeros(LC), np.arange(nframes - LC)])[:, np.newaxis]

    f = np.cumsum(features, 0)
    s = np.cumsum(features ** 2, 0)

    f = (np.r_[f[RC:], np.repeat(f[[-1]], RC, axis=0)] -
         np.r_[np.zeros((LC + 1, dim)), f[:-LC - 1]]) / n

    s = (np.r_[s[RC:], np.repeat(s[[-1]], RC, axis=0)] -
         np.r_[np.zeros((LC + 1, dim)), s[:-LC - 1]]) / \
        (n - 1 if unbiased else n) - f ** 2 * (n / (n - 1) if unbiased else 1)

    std = np.sqrt(np.maximum(s, 1e-12))

    normalised_features = (features - f) / std
    normalised_features[s == 0] = 0

    return normalised_features