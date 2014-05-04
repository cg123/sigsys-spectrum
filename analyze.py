#!/usr/bin/env python3
# 4/15/2014
# Charles O. Goddard

import numpy
from numpy.fft import fft, fftfreq, ifft
from scipy.io import wavfile
from scipy.signal import find_peaks_cwt
import scipy.optimize
import matplotlib.pyplot as plt

import sys

def todb(x):
    return 20 * numpy.log10(abs(x))


def smooth(x,window_len=11,window=numpy.hanning):
    s=numpy.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    w = window(window_len)
    y=numpy.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]


def peak(omega_0, k, omega):
    return (k/(2*numpy.pi)) / ((omega - (numpy.zeros(omega.shape) + omega_0))**2 + (k/2)**2)


def make_response(omega, O, K, A):
    return sum(peak(O[i], K[i], omega) * A[i] for i in range(len(K)))


def main(fn):
    (fs, x) = wavfile.read(fn)
    print(fs, x.shape)

    x = numpy.array(x, dtype=float) / 2.0**16
    print('max(x) =', max(abs(x)))
    shape = smooth(abs(x), window_len = 0.01 * fs)

    window = numpy.kaiser(len(x), beta=14)
    Xw = fft(x * window / shape, n=32768)
    aXw = abs(Xw)
    omega = fftfreq(Xw.size, 1./fs)

    # for omega, ax in zip(omega, aXw):
    #     print(omega, ', ', ax)
    # print

    plt.figure(1)
    plt.semilogx(omega, todb(Xw))
    # idx = numpy.argmax(aXw)
    idx = find_peaks_cwt(aXw, numpy.array([4, 8, 12, 16, 24, 32], dtype=float))
    idx.sort(key=lambda idx: aXw[idx], reverse=True)
    idx = [i for i in idx if omega[i] > 0]

    for i in range(len(idx)):
        while aXw[idx[i]-1] > aXw[idx[i]] or aXw[idx[i]+1] > aXw[idx[i]]:
            if aXw[idx[i]-1] > aXw[idx[i]]:
                idx[i] -= 1
            else:
                idx[i] += 1

    print('Detected frequency peaks:')
    for i in range(len(idx[:50])):
        print('{0}. {1} Hz ({2} db, {3}*f_0)'.format(i+1, omega[idx[i]], todb(aXw[idx[i]]), omega[idx[i]]/omega[idx[0]]))

    O = omega[idx[:50]]
    K = numpy.zeros(len(O)) + 1
    A = aXw[idx[:50]]

    K, _ = scipy.optimize.leastsq(lambda K: make_response(omega, O, K, A) - aXw, K)
    # O = OKA[0:len(O)]
    # K = OKA[len(O):2*len(O)]
    # A = OKA[2*len(O):]
    print(O, K, A)

    omega_1 = fftfreq(len(x), 1./fs)
    resp = make_response(omega_1, O, K, A)

    plt.hold(True)
    plt.semilogx(omega_1, todb(resp))
    plt.semilogx(omega[idx[0]], todb(Xw[idx[0]]), 'go')

    plt.legend(["Original FFT Amplitude", "Synthesized Response", "Fundamental Frequency"])
    plt.show()

    note = 440  # Hz

    res = numpy.zeros(len(x), dtype=complex)
    t = numpy.linspace(0, len(x)/fs, len(x))
    for note in [440]:#[220, 293.66, 440, 698.46]:
        for i in range(len(O)):
            freq = O[i]/O[0] * note
            res += A[i] * numpy.exp(1j*(-2*numpy.pi*t*freq) - K[i]*abs(t)) #numpy.cos(freq * 2 * numpy.pi * t)  # numpy.exp(1j*(-2*numpy.pi*t*freq) - K[i]*abs(t))#

    res = numpy.real(res)
    res *= shape
    res /= max(res)
    # res = numpy.real(ifft(resp))
    # res *= smooth(abs(x), window_len = 0.01 * fs) / smooth(abs(res), window_len = 0.01 * fs)

    wavfile.write('out.wav', fs, res)
    print('Wrote out.wav')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
