from os import supports_effective_ids
from librosa.core import audio
import numpy
import soundcard as sc
import librosa
import librosa.display
import matplotlib.pyplot as plt
import asyncio
import statistics

Fs = 44100 # sample rate
T = 1/Fs # sampling period
t = 0.1 # seconds of sampling
N: int =  Fs*t # total points in signal

lb = sc.all_microphones(include_loopback=True)[0]

frequencies = librosa.core.fft_frequencies(n_fft=2048*4)
freq_index_ratio = len(frequencies)/frequencies[len(frequencies)-1] / 2

currentlyplaying = 0
subbass = bass = lowmid = mid = uppermid = presence = brilliance = 0
basshit = 0
avgbass = avgmid = avghigh = 0

def main() -> int:
    while (True):
        spectrogram = audioanalysis()

        if (spectrogram.ndim == 2 ):
            spectrogram = spectrogram.flatten('F')
            spectrogram = spectrogram[:int(spectrogram.size/2)]
        
        currentlyplaying = 0
        for x in range(2,4000):
            if (spectrogram[x] != 0):
                currentlyplaying = 1
        
        subbass = (7+(max(min(statistics.mean([spectrogram[int(20*freq_index_ratio)], 
                                   spectrogram[int(40*freq_index_ratio)], 
                                   spectrogram[int(60*freq_index_ratio)]]), 0), -70)/10))/7 * currentlyplaying

        bass = (7+(max(min(statistics.mean(   [spectrogram[int(80*freq_index_ratio)], 
                                   spectrogram[int(120*freq_index_ratio)], 
                                   spectrogram[int(160*freq_index_ratio)], 
                                   spectrogram[int(200*freq_index_ratio)]]) , 0), -70)/10))/7 * currentlyplaying

        lowmid = (7+(max(min(statistics.mean( [spectrogram[int(300*freq_index_ratio)], 
                                   spectrogram[int(400*freq_index_ratio)], 
                                   spectrogram[int(500*freq_index_ratio)]]) , 0), -70)/10))/7 * currentlyplaying

        mid = (7+(max(min(statistics.mean(    [spectrogram[int(500*freq_index_ratio)], 
                                   spectrogram[int(800*freq_index_ratio)], 
                                   spectrogram[int(1400*freq_index_ratio)], 
                                   spectrogram[int(2000*freq_index_ratio)]]) , 0), -70)/10))/7 * currentlyplaying

        uppermid = (7+(max(min(statistics.mean([spectrogram[int(2000*freq_index_ratio)], 
                                   spectrogram[int(2700*freq_index_ratio)], 
                                   spectrogram[int(3300*freq_index_ratio)], 
                                   spectrogram[int(4000*freq_index_ratio)]]) , 0), -70)/10))/7 * currentlyplaying

        presence = (7+(max(min(statistics.mean([spectrogram[int(4000*freq_index_ratio)], 
                                   spectrogram[int(4700*freq_index_ratio)], 
                                   spectrogram[int(5300*freq_index_ratio)], 
                                   spectrogram[int(6000*freq_index_ratio)]]) , 0), -70)/10))/7 * currentlyplaying

        brilliance = (7+(max(min(statistics.mean([spectrogram[int(6000*freq_index_ratio)], 
                                   spectrogram[int(8000*freq_index_ratio)], 
                                   spectrogram[int(14000*freq_index_ratio)], 
                                   spectrogram[int(20000*freq_index_ratio)]]) , 0), -70)/10))/7 * currentlyplaying

        avgbass = statistics.mean([subbass, bass])
        avgmid = statistics.mean([lowmid, mid, uppermid])
        avghigh = statistics.mean([presence, brilliance])

        print([avgbass, avgmid, avghigh])

    return 0

def audioanalysis():
    with lb.recorder(samplerate=Fs) as mic:
        rawdata = mic.record(numframes=2205)
        datalen: int = int(rawdata.size/2)
        monodata = numpy.empty(datalen)
        for x in range(0, datalen):
            monodata[x] = max(rawdata[x][0], rawdata[x][1])
        data = numpy.abs(librosa.stft(monodata, n_fft=2048*4))
        return librosa.amplitude_to_db(data, ref=numpy.max)
        

if __name__ == "__main__":
    main()