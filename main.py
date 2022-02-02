import sounddevice as sd
import soundfile as sf
import librosa
import tensorflow
from keras.models import load_model
import numpy as np
import time

TARGET_SR = 8000
""" print('starting...')

samplerate = 16000  
duration = 1 # seconds
filename = 'stop.wav'
print("say a word in 3")
time.sleep(1)
print('...2')
time.sleep(1)
print('...1')
time.sleep(1)
print('NOW!!!')

mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
    channels=1, blocking=True)
print("end")
sd.wait()
sf.write(filename, mydata, samplerate) """

samples, sample_rate = librosa.load('commands/go_1.wav', sr=1600)
sample = librosa.resample(samples, sample_rate, TARGET_SR)
diff = TARGET_SR - sample.shape[0]

if diff > 0:
    sample = np.concatenate([sample, np.zeros(diff)])
elif diff < 0:
    sample = sample[:8000]

model = load_model('speechrec2.hdf5')
classes = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']

def predict(audio):
    prob=model.predict(audio.reshape(1, 8000,1))
    index=np.argmax(prob[0])
    return classes[index]

print('SAMPLE RATE', sample.shape[0])
print(predict(sample))
print('all done...')