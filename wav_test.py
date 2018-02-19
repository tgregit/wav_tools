import wav_tools as wt
from scipy.fftpack import fft, dct, idct

path_to_vctk_data = '/home/foo/data/VCTK-Corpus'
vctk = wt.vctk(path_to_vctk_data)

wav = vctk.get_wav(228,2)  # speaker #255, sentence #2


# print(wav.shape[0])
# dct = dct(wav) / 20.0
# print(dct[0:100])

window_length = 1024
coef_to_keep = 170

d = wt.wav_to_dct_coef(wav, window_length, coef_to_keep)
print(d)



#print(vc.data_path)
# wt.foo()