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

    def strip_silence(self, my_data, my_window_length):
        total_windows = my_data.shape[0] // my_window_length
        start_index = 0
        end_index = my_data.shape[0]

        std_threshold = .008

        for i in range(0, total_windows):
            current_window = my_data[i*my_window_length:(i*my_window_length) + my_window_length]
            if current_window.std() > std_threshold:
                start_index = i * my_window_length
                break

        for i in range(total_windows - 1, 0, -1):
            current_window = my_data[i * my_window_length:(i * my_window_length) + my_window_length]
            print(current_window.std())
            if current_window.std() > std_threshold:
                end_index = (i * my_window_length)
                break
        data_silence_stripped = my_data[start_index:end_index]

        print(start_index,end_index)

        return data_silence_stripped



    def get_wav(self, speaker_id, sentence_id, strip_silence=False):
        speaker_prefix = 'p' + str(speaker_id)
        number_string = self.get_number_string_from_int(sentence_id)
        full_wav_path = self.data_path + '/wav48/' + speaker_prefix + '/' + speaker_prefix + '_' + number_string + '.wav'

        data, sr = librosa.load(full_wav_path, self.sample_rate)

        if strip_silence == True:
            data = self.strip_silence(data, 2048)

            print('strip silence here')

        return data


def normalize(my_data):
    my_data = (my_data * 1.0) - (my_data.min() * 1.0)
    my_data = (my_data * 1.0) / (my_data.max() * 1.0)

    return my_data


def normalize_neg_one(my_data):
    my_data = my_data - (my_data.min() * 1.0)
    my_data = my_data / ((my_data.max()) * 1.0)
    my_data = my_data - .5
    my_data = my_data * 2.0

    return my_data


def wav_to_dct_coef_compressed(my_wav, my_window_length, my_coef_to_keep):
    total_windows = my_wav.shape[0] // my_window_length
    compressed_dct = np.zeros((total_windows, my_coef_to_keep))

    my_wav = my_wav[0:total_windows * my_window_length]
    my_wav = np.reshape(my_wav,(total_windows, my_window_length))
    wav_dct = dct(my_wav, norm='ortho')

    for i in range(0, total_windows - 1):
        full_dct = wav_dct[i]
        compressed_dct[i] = full_dct[0: my_coef_to_keep]

    return compressed_dct


def dct_coef_compressed_to_amplitudes(my_dct_compressed, my_window_length):
    total_windows = my_dct_compressed.shape[0]
    coef_keep = my_dct_compressed.shape[1]
    extra_padding = my_window_length - coef_keep

    dct_padded = np.zeros((total_windows, my_window_length))

    for i in range(0, total_windows):
        compressed_row = my_dct_compressed[i]
        dct_padded[i][0: coef_keep] = compressed_row   # np.zeros has already padded the ends with 0.0's

    inverse_dct = idct(dct_padded, norm='ortho') # these are now amplitudes

    amplitudes_1_d = np.reshape(inverse_dct, (total_windows * my_window_length))


    return amplitudes_1_d



