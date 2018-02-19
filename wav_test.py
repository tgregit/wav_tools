import wav_tools as wt
from scipy.fftpack import fft, dct, idct
import librosa
import numpy as np
window_length = 1024
coef_to_keep = 256#128#512#256

vctk = wt.vctk('/home/foo/data/VCTK-Corpus')

wav = vctk.get_wav(228, 2, True)  # speaker #255, sentence #2  #wav = wt.normalize(wav) T

librosa.output.write_wav('/home/foo/data/wavs-test/28.wav', wav, sr=vctk.sample_rate)

# dct_compressed = wt.wav_to_dct_coef_compressed(wav, window_length, coef_to_keep)
# for dc in dct_compressed:
#     print(dc)
# i = wt.dct_coef_compressed_to_amplitudes(dct_compressed, window_length)
# print(i, i.shape)
#
# librosa.output.write_wav('/home/foo/data/wavs-test/28idct.wav', i, sr=vctk.sample_rate)
