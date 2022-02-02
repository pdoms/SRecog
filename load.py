import librosa

samples, sample_rate = librosa.load('left.wav', sr = 1600)
print(samples.shape, sample_rate)
