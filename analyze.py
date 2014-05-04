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
import pickle
import os.path

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
    return sum(peak(O[i], max(0, K[i]), omega) * A[i] for i in range(len(K)))


def extract_parameters(fn, debug=False):
    (fs, x) = wavfile.read(fn)
    if debug:
        print('f_s:', fs, 'shape:', x.shape)
    if len(x.shape) > 1:
        x = sum(x[:,i] for i in range(x.shape[1]))

    x = numpy.array(x, dtype=float) / 2.0**16
    if debug:
        print('max(x) =', max(abs(x)))
    shape = smooth(abs(x), window_len = 0.1 * fs)

    window = numpy.kaiser(len(x), beta=14)
    Xw = fft(x * window, n=32768)
    aXw = abs(Xw)
    omega = fftfreq(Xw.size, 1./fs)

    idx = find_peaks_cwt(aXw, numpy.array([4, 8, 12, 16, 24, 32], dtype=float))

    for i in range(len(idx)):
        while True:
            if idx[i] > 0 and aXw[idx[i]-1] > aXw[idx[i]]:
                idx[i] -= 1
            elif idx[i] < len(aXw)-1 and aXw[idx[i]+1] > aXw[idx[i]]:
                idx[i] += 1
            else:
                break

    idx.sort(key=lambda idx: aXw[idx], reverse=True)
    idx = [i for i in idx if omega[i] > 1]

    idx_f0 = idx[numpy.argmax(-omega[idx[:5]])]
    if debug:
        print('f0:', omega[idx_f0])

    idx = [idx_f0] + [i for i in idx if i != idx_f0]

    if debug:
        print('Detected frequency peaks:')
        for i in range(len(idx[:50])):
            print('{0}. {1} Hz ({2} db, {3}*f_0)'.format(i+1, omega[idx[i]], todb(aXw[idx[i]]), omega[idx[i]]/omega[idx[0]]))

    O = omega[idx[:50]]
    K = numpy.zeros(len(O)) + 1
    A = aXw[idx[:50]]

    K, _ = scipy.optimize.leastsq(lambda K: make_response(omega, O, K, A) - aXw, K)
    if debug:
        print(K)

    return O, K, A, shape


def synthesize(f0, shape, O, K, A, fs=44100):
    res = numpy.zeros(len(shape), dtype=complex)
    t = numpy.linspace(0, len(shape)/fs, len(shape))
    for i in range(len(O)):
        freq = O[i]/O[0] * f0
        res += A[i] * numpy.exp(1j*(-2*numpy.pi*t*freq) - K[i]*abs(t))

    res = numpy.real(res)
    res *= shape
    res /= max(res)
    return res


def main(argv):
    if not argv:
        print('usage:\n\tanalyze.py command [options]\n\ncommands:\n\tmodel <sample.wav> [debug]\n\tplay <model.dat> <freq>')
        return -1

    command = argv[0]
    if command == 'model':
        debug = False
        if len(argv) not in (2, 3):
            print('Expected one or two arguments')
            return -1
        if len(argv) == 3:
            if argv[2] == 'debug':
                debug = True
            else:
                print('Third argument to model must be debug')
                return -1

        O, K, A, shape = extract_parameters(argv[1], debug)

        fn = os.path.basename(argv[1]).replace('.wav','')
        pickle.dump((O, K, A, shape), open('models/'+fn+'.dat', 'wb'))
        return 0
    elif command == 'play':
        assert(len(argv) == 3)
        O, K, A, shape = pickle.load(open(argv[1], 'rb'))
        note = int(argv[2])
        wave = synthesize(note, shape, O, K, A)
        new_fn = os.path.basename(argv[1]).replace('.dat','')+'-'+str(note)+'.wav'
        wavfile.write(new_fn, 44100, wave)
    else:
        print('Invalid command')
        return -1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
