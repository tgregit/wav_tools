import librosa
import numpy as np
from scipy.fftpack import fft, dct, idct


class vctk:
    data_path = './'
    sample_rate = 22050

    def __init__(self, data_path):
        self.data_path = data_path

    def get_number_string_from_int(self, my_int):
        length = len(str(my_int))
        zeros_to_prepend = 3 - length
        zeros = '0' * zeros_to_prepend
        num_string = zeros + str(my_int)

        return num_string

    def get_wav(self, speaker_id, sentence_id, strip_silence=False):
        speaker_prefix = 'p' + str(speaker_id)
        number_string = self.get_number_string_from_int(sentence_id)
        full_wav_path = self.data_path + '/wav48/' + speaker_prefix + '/' + speaker_prefix + '_' + number_string + '.wav'

        data, sr = librosa.load(full_wav_path, self.sample_rate)

        if strip_silence:
            print('strip silence here')

        return data


def wav_to_dct_coef(my_wav, my_window_length, my_coef_to_keep):
    total_windows  = my_wav.shape[0] // my_window_length
    dct_coef = np.zeros((total_windows, my_coef_to_keep), dtype=np.float)

    for i in range(0, total_windows):
        window = my_wav[i*my_window_length:((i*my_window_length) + my_window_length)]
        window_dct = dct(window)
        dct_coef[i] = window_dct[0:my_coef_to_keep]

    return dct_coef





def foo():
    print('foo')