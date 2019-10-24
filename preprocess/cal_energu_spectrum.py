import sys
import os
import numpy as np
import librosa
from scipy import signal
import soundfile
src_dir = sys.argv[1]
tar_dir = sys.argv[2]

def cal_all_energy(path):
    (audio, fs) = soundfile.read(path)
    ham_win = np.hamming(512)
    [f, t, x] = signal.spectral.spectrogram(
        audio, fs = 16000,
        windiw = ham_win,
        nperseg = 512,
        noverlap=352,
        detrend=False,
        return_onesided=True,
        model='complex'
    )

def cal_band_energy(path):
    (audio, fs) = soundfile.read(path)
    melspec = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=512, hop_length=160, n_mels=80)
    logspec = librosa.logamplitude(melspec)
    
def traverse(wav_dir, save_dir):
    for fi in os.listdir(wav_dir):
        wav_path = os.path.join(wav_dir, fi)
        log_mel = cal_band_energy(wav_path)
        wav_name = fi.split('.wav')
        wav_dir = os.path.join(save_dir, '%s.npy'%wav_name)

if __name__=='__main__':
    traverse(src_dir, tar_dir)
