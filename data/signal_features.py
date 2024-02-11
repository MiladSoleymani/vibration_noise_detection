import numpy as np
import scipy.signal as ss
import scipy.stats as sp


def psd(signal, fs, nperseg=1024):
    f, Pxx_den = ss.welch(signal, fs, nperseg=nperseg)
    return np.log(Pxx_den)


def static_features(signal):
    def minimum(c):
        return np.min(c, axis=1)

    def maximum(c):
        return np.max(c, axis=1)

    def mean(c):
        return np.mean(c, axis=1)

    def std(c):
        return np.std(c, axis=1)

    def coeff_var(c):
        return np.log(sp.variation(c, axis=1))

    def kurtosis(c):
        return np.log(sp.kurtosis(c, axis=1))

    def wavelet_packet_energy(signal):
        return np.log(np.sqrt(np.sum(np.power(signal, 2) / signal.shape[1], axis=1)))

    feats = [minimum, maximum, mean, std, kurtosis, wavelet_packet_energy]  # coeff_var

    return np.array([feat(signal) for feat in feats]).T
