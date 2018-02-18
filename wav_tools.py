import librosa
import numpy as np


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


    def get_wav(self, speaker_id, sentence_id):
        speaker_prefix = 'p' + str(speaker_id)

        number_string = self.get_number_string_from_int(sentence_id)

        full_wav_path = self.data_path + '/wav48/' + speaker_prefix + '/' + speaker_prefix + '_' + number_string

        print(full_wav_path)


def foo():
    print('foo')