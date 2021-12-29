import numpy
import soundcard as sc
import librosa
import librosa.display
import matplotlib.pyplot as plt

lb = sc.all_microphones(include_loopback=True)[0]

Fs = 44100 # sample rate
T = 1/Fs # sampling period
t = 0.1 # seconds of sampling
N: int =  Fs*t # total points in signal

with lb.recorder(samplerate=Fs) as mic:
    while True:
    #for _ in range(0,100):
        rawdata = mic.record(numframes=2205)
        datalen: int = int(rawdata.size/2)
        monodata = numpy.empty(datalen)
        for x in range(0, datalen):
            monodata[x] = max(rawdata[x][0], rawdata[x][1])
        data = numpy.abs(librosa.stft(monodata, n_fft=2048*4))

        spectrogram = librosa.amplitude_to_db(data, ref=numpy.max)

        frequencies = librosa.core.fft_frequencies(n_fft=2048*4)
        freq_index_ratio = len(frequencies)/frequencies[len(frequencies)-1]

        

        librosa.display.specshow( spectrogram,
                         y_axis='log', x_axis='time')
        plt.title('spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()

        quit()
        monodata = [0] * rawdata.size
        for x in range(0, rawdata.size):
            monodata[x] = max(rawdata[x][0], rawdata[x][1])
        data = numpy.fft.fft(monodata)[0:int(N/2)]/N
        data[1:] = 2 * data[1:]
        realdata = numpy.abs(data)
        print(realdata)

        f = Fs * numpy.arange((N/2))/N # freq vector
        #print(f)
        