import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
import csv
from scipy import signal
from scipy.signal import blackman
from scipy.signal import butter, lfilter
from array import *

Time = []
Voltage = []


with open('FDSM_out_60s_3.38ms_4u.csv', 'r') as csvfile:
    heading = next(csvfile)
    FDSM_out_dff_unbuffered = csv.reader(csvfile, delimiter = ',')

    for row in FDSM_out_dff_unbuffered:
        #klippe bort dataen før systemet starter
        if row[0] >= '0.0001':
            Time.append(float(row[0]))
            Voltage.append(float(row[1]))

y = np.array(Voltage)
x = np.array(Time)
#print(y.shape)

timestep = x[1] - x[0]
#print(timestep)

N = int(y.shape[0])
T = timestep

xf = fftfreq(N, T)[:N//2]
#Remove DC component
xf = xf[1:]
#Limit to band of interest
xf = xf[0:660]

yf = fft(y)
#only positive frequency
FFT_yf = yf[0:N//2] 
FFT_yf = FFT_yf[1:]
FFT_yf = FFT_yf[0:660]
#scaling of y axies of the FFT
FFT_yf1 = (2/N * np.abs(FFT_yf))*4
'''
sos = signal.butter(2, 2, 'lp', fs=1/T, output='sos')
filtered = signal.sosfilt(sos, y)
plt.plot(Time, filtered)
plt.show()
'''
#calculate power of the signal
def power_of_signal(x):
    N = x.shape[0]
    return np.sum(np.power(x, 2)) / N

f_signal = []
signal_signal = []
f_noise = []
signal_noise = []

#Distinguish input signal with the rest 
for x, y in zip(xf, FFT_yf1):
    if y >= 0.02:
        f_signal.append(x)
        signal_signal.append(y)
    else:
        f_noise.append(x)
        signal_noise.append(y)

s_signal_signal = np.array(signal_signal)
a_signal_noise = np.array(signal_noise)

print('the length of the x coordinates of the freq is', len(f_noise))
print('the length of the y coordinates of the freq is', len(signal_noise))
print(f_signal, signal_signal)
'''
#Define Band of interest (BOI)
f_noise_BOI = []
signal_noise_BOI = []

for k in f_noise:
    if k <= 11:
        f_noise_BOI.append(k) 

for x,y in zip(signal_noise, f_noise_BOI):
    signal_noise_BOI.append(x)

print(len(f_noise_BOI), len(signal_noise_BOI))
'''
#calculation of SQNR with all hamonics
all_harmonics_y = []
all_harmonics_x = []

for n in range(11):
    for x,y in zip(xf, FFT_yf1):
        if x == n*f_signal[0]:
            all_harmonics_y.append(y)
            all_harmonics_x.append(x)
a_all_harmonics_y = np.array(all_harmonics_y)

#remove fundamental signal and harmonic signals from the sample
FFT_yf1_list = FFT_yf1.tolist()
noise = []
for i in FFT_yf1_list:
    noise.append(i)

for k in all_harmonics_y:
    noise.remove(k)
    
print('length of the noise is',len(noise))
    
a_noise = np.array(noise)

print('length of harmonics is', len(all_harmonics_x))
#print(all_harmonics_x[0:15])

SQNR_allharmonics =10*np.log10(power_of_signal(s_signal_signal) / power_of_signal(a_noise))
SINAD = 10*np.log10(power_of_signal(s_signal_signal) / power_of_signal(a_signal_noise))
THD = 10*np.log10(power_of_signal(s_signal_signal) / power_of_signal(a_all_harmonics_y))
print(10*np.log10(power_of_signal(s_signal_signal)), 10*np.log10(power_of_signal(a_noise)))
print('SQNR is', SQNR_allharmonics,'dB')
print('SINAD', 10*np.log10(power_of_signal(s_signal_signal)), 10*np.log10(power_of_signal(a_signal_noise)))
print('SINAD is', SINAD,'dB')
print('THD is', THD)
print('amplitude of the signal is', s_signal_signal)
print('length of the noise is 6059', a_signal_noise.shape)

plt.plot(xf, FFT_yf1)
plt.grid()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Signal')
plt.show()

plt.semilogx(xf, 20*np.log10(FFT_yf1))
plt.text(10, -28, 'SQNR = 35.83 dB')
plt.text(10, -38, 'SINAD = 35.86 dB')
plt.text(10, -48, 'THD = 19.34 dB')
plt.xlabel('log frequency')
plt.ylabel('Signal in dB')
plt.grid()
plt.show()




