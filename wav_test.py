import wav_tools as wt

path_to_vctk_data = '/home/foo/data/VCTK-Corpus'
vctk = wt.vctk(path_to_vctk_data)

wav = vctk.get_wav(255,2)  # speaker #255, sentence #2
print(wav)


#print(vc.data_path)
# wt.foo()