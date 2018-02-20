import wav_tools as wt
from scipy.fftpack import fft, dct, idct
import librosa
import cv2

import numpy as np
window_length = 512
coef_to_keep = 168#32#128

vctk = wt.vctk('/home/foo/data/VCTK-Corpus')

global_max = 1.0#5.0#2.0


for speaker_id in range(258,259):
    for sentence_id in range(43,44):

        wav = vctk.get_wav(speaker_id, sentence_id, False)  # speaker #255, sentence #2  #wav = wt.normalize(wav)

        librosa.output.write_wav('/home/foo/data/wavs-test/28.wav', wav, sr=22050)

        dct_compressed = wt.wav_to_dct_coef_compressed(wav, window_length, coef_to_keep)
        for d in dct_compressed:
            for i in range(0, d.shape[0]):
                if d[i] > global_max:
                    d[i] = global_max
                if d[i] < (-1.0 * global_max):
                    d[i] = (-1.0 * global_max)


        #dct_compressed = wt.normalize(dct_compressed)
        i = wt.dct_coef_compressed_to_amplitudes(dct_compressed, window_length)
        librosa.output.write_wav('/home/foo/data/wavs-test/28inverse-capped.wav', i, sr=vctk.sample_rate)

        dct_compressed = dct_compressed * 255.0
        file_name = '/home/foo/data/wavs-test/vctk-' + str(speaker_id) + '-' + str(sentence_id) + '.png'
        cv2.imwrite(file_name, dct_compressed)



#normalize wuith dixed max and min, ~~2.0????

# i = wt.dct_coef_compressed_to_amplitudes(dct_compressed, window_length)
# print(i, i.shape)
#
# librosa.output.write_wav('/home/foo/data/wavs-test/28idct.wav', i, sr=vctk.sample_rate)
