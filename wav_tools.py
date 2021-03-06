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
        ignored_samples = end_index - (total_windows * my_window_length)
        print(ignored_samples)

        std_threshold = .05

        for i in range(0, total_windows):
            current_window = my_data[i * my_window_length:(i * my_window_length) + my_window_length]

        for i in range(0, total_windows):
            current_window = my_data[i*my_window_length:(i * my_window_length) + my_window_length]
            if current_window.max() > std_threshold:
                start_index = i * my_window_length
                break

        data_rev = my_data.copy()   # TODO: REFACTOR THIS, THE K loop below is identical to the i, pull out to function
                                    # OR Recombine into single loop
        for j in range(0, data_rev.shape[0]):
            data_rev[j] = my_data[data_rev.shape[0] - j - 1]

        for k in range(0, total_windows):
            current_window = data_rev[k*my_window_length:(k * my_window_length) + my_window_length]
            if current_window.max() > std_threshold:
                end_index = ((my_window_length * total_windows) - (k * my_window_length)) + ignored_samples
                break

        data_silence_stripped = my_data[start_index:end_index]

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


def get_non_linear_quantization_values(my_quantizations, my_percent):
    half_quants = my_quantizations // 2
    quant_values = []

    amp_space_remaining = 1.0
    for i in range(0, half_quants):
        val = amp_space_remaining * my_percent
        quant_values.append(val)
        amp_space_remaining = amp_space_remaining - (amp_space_remaining - val)

    quant_values[half_quants - 1] = 0.0

    return quant_values

def linear_to_non_linear_integer(my_data, my_quantizations, my_percent):
    rows = my_data.shape[0]
    columns = my_data.shape[1]
    my_data = np.reshape(my_data,(rows * columns))
    half_quantizations = my_quantizations // 2
    quantization_values = get_non_linear_quantization_values(my_quantizations, my_percent)
    data_integers = []
    count = np.zeros((my_quantizations))

    for i in range(0, my_data.shape[0]):
        sign = 1
        if my_data[i] < 0.0:
            sign = -1
        val = abs(my_data[i])
        for k in range(0, half_quantizations):
            q_val = quantization_values[k]

            if val > q_val or val == q_val:
                index = k
                if sign == 1:
                    index = my_quantizations - k - 1
                data_integers.append(index)
                count[index] = count[index] + 1.0
                break
    data_return = np.array(data_integers, dtype=np.int)
    data_return = np.reshape(data_return,(rows,columns))
    return data_return

def non_linear_integer_to_linear(my_data_ints, my_quantizations, my_percent):
    rows = my_data_ints.shape[0]
    columns = my_data_ints.shape[1]
    my_data_ints = np.reshape(my_data_ints, (rows * columns))
    half_quantizations = my_quantizations // 2

    quantization_values = get_non_linear_quantization_values(my_quantizations, my_percent)

    linear_amps = np.zeros((my_data_ints.shape[0]), dtype=np.float)

    for i in range(0, my_data_ints.shape[0]):
        sign = -1

        index = my_data_ints[i]
        if index >= half_quantizations:
            index = my_quantizations - index - 1#index - half_quantizations
            sign = 1
            #print(index)
        linear_value = quantization_values[index]
        linear_value = linear_value * sign * 1.0
        linear_amps[i] = linear_value

    linear_amps = np.reshape(linear_amps, (rows, columns))
    return linear_amps

def wav_to_dct_coef_compressed(my_wav, my_window_length, my_coef_to_keep, my_cap_value):
    total_windows = my_wav.shape[0] // my_window_length
    compressed_dct = np.zeros((total_windows, my_coef_to_keep))

    my_wav = my_wav[0:total_windows * my_window_length]
    my_wav = np.reshape(my_wav,(total_windows, my_window_length))
    wav_dct = dct(my_wav, norm='ortho')
    for r in wav_dct:
        for k in range(0, r.shape[0]):
            if r[k] > my_cap_value:
                r[k] = my_cap_value
            if r[k] < (-1.0 * my_cap_value):
                r[k] = (-1.0 * my_cap_value)

    for i in range(0, total_windows ):
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



