import librosa
from scipy.spatial.distance import euclidean
import numpy as np
import math




ls = [1,2,3,4,5]
ls1 = [10,45,68,45,90]
print('EC1', euclidean(ls, ls1))

def euclidean_distance(v,w):
    return math.sqrt(sum(math.pow(v_i - w_i, 2) for v_i, w_i in zip(v,w)))

print('EC2', euclidean_distance(ls, ls1))






sr = 16000
samples_1, sample_rate_1 = librosa.load('from_ds/0bd689d7_nohash_1.wav', sr)

samples_2, sample_rate_2 = librosa.load('stop.wav', sr)
print(samples_1.shape)
print(samples_2.shape)
eucl = euclidean(samples_1, samples_2)
print(eucl)


