import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy.io
from scipy import signal

#oppgave1a
def konvin3190(h, x, ylen=1):
    M = h.size
    N = x.size

    if ylen == 0:
        y = np.zeros(N)
        for m in range(1, M+1):
            for n in range(1, N-M+1):
                k = n + m - 1
                y[k-1] = y[k-1] + (h[m-1]*x[n-1])
        return(y)
    elif ylen==1:
        L = (N + M - 1)
        y = np.zeros(L)
        for m in range(1, M+1):
            for n in range(1, N+1):
                k = n + m - 1
                y[k-1] = y[k-1] + (h[m-1]*x[n-1])
        return(y)

#opggave1b
def frekspekin3190(x, N, fs):
    M = x.size
    f_omega= np.linspace(-0.5, 0.5, N)
    omega = (2*np.pi)*(f_omega)
    f = fs*f_omega
    X = np.zeros(N, dtype=complex)
    for n in range(N):
        for m in range(M):
            X[n] = X[n] + (x[m]*np.exp(-1j*omega[n]*m))
    return(f, X)

#oppgave1c
def test():
    t = 5 #sekunder
    fs = 100
    N = fs*t
    t_ary = np.linspace(0,t,N)
    f1 = 10 #Hz
    f2 = 20 #Hz
    x = np.sin(2*np.pi*f1*t_ary) + np.sin(2*np.pi*f2*t_ary)
    h = np.ones(5)/5

    plt.plot(frekspekin3190(x,N,fs)[0],abs(frekspekin3190(x,N,fs)[1]))
    plt.title(r'$x[n] = sin(2 \pi f_1 t) + sin(2 \pi f_2 t) $')
    plt.xlabel('Tid i sekunder')
    plt.ylabel('Utslag')
    plt.show()

    plt.plot(frekspekin3190(x,N,fs)[0], abs(frekspekin3190(x,N,fs)[1]))
    plt.title(r'$X[e^{j \omega}] $')
    plt.xlabel('Frekvens')
    plt.ylabel('Utslag')
    plt.show()

    plt.plot(frekspekin3190(h,N,50)[0], abs(frekspekin3190(h,N,50)[1]))
    plt.title(r'$H[e^{j \omega}] $')
    plt.xlabel('Frekvens')
    plt.ylabel('Utslag')
    plt.show()

    Y = konvin3190(h,x,0)
    plt.plot(frekspekin3190(Y,N,fs)[0], abs(frekspekin3190(Y,N,fs)[1]))
    plt.title(r'$Y[e^{j \omega}] $')
    plt.xlabel('Frekvens')
    plt.ylabel('Utslag')
    plt.show()

test()


#oppgave2a
seismisk_data = scipy.io.loadmat('yinic.mat')
seismogram1 = ((seismisk_data['seismogram1']))
seismogram2 = ((seismisk_data['seismogram2']))
t = (seismisk_data['t'])
offset1 = (seismisk_data['offset1'])[0]
offset2 = (seismisk_data['offset2'])[0]

h1 = np.array([0.0002, 0.0001, -0.0001, -0.0005, -0.0011, -0.0017, -0.0019,
    -0.0016, -0.0005, 0.0015, 0.0040, 0.0064, 0.0079, 0.0075, 0.0046,
    -0.0009, -0.0084, -0.0164, -0.0227, -0.0248, -0.0203, -0.0079,
    0.0127, 0.0400, 0.0712, 0.1021, 0.1284, 0.1461, 0.1523, 0.1461,
    0.1284, 0.1021, 0.0712, 0.0400, 0.0127, -0.0079, -0.0203, -0.0248,
    -0.0227, -0.0164, -0.0084, -0.0009, 0.0046, 0.0075, 0.0079, 0.0064,
    0.0040, 0.0015, -0.0005, -0.0016, -0.0019, -0.0017, -0.0011,
    -0.0005, -0.0001, 0.0001, 0.0002])

h2 = np.array([-0.0002, -0.0001, 0.0003, 0.0005, -0.0001, -0.0009, -0.0007,
    0.0007, 0.0018, 0.0005, -0.0021, -0.0027, 0.0004, 0.0042, 0.0031,
    -0.0028, -0.0067, -0.0023, 0.0069, 0.0091, -0.0010, -0.0127,
    -0.0100, 0.0077, 0.0198, 0.0075, -0.0193, -0.0272, 0.0014, 0.0386,
    0.0338, -0.0246, -0.0771, -0.0384, 0.1128, 0.2929, 0.3734, 0.2929,
    0.1128, -0.0384, -0.0771, -0.0246, 0.0338, 0.0386, 0.0014, -0.0272,
    -0.0193, 0.0075, 0.0198, 0.0077, -0.0100, -0.0127, -0.0010, 0.0091,
    0.0069, -0.0023, -0.0067, -0.0028, 0.0031, 0.0042, 0.0004, -0.0027,
    -0.0021, 0.0005, 0.0018, 0.0007, -0.0007, -0.0009, -0.0001, 0.0005,
    0.0003, -0.0001, -0.0002])

def _seismogram():
    plt.imshow(seismisk_data['seismogram1'], cmap='gray', aspect = 'auto',
    extent=[offset1[0], offset1[-1], t[-1], t[0]])
    plt.colorbar()
    plt.show()

    plt.imshow(seismisk_data['seismogram2'], cmap='gray', aspect = 'auto',
    extent=[offset2[0], offset2[-1], t[-1], t[0]])
    plt.colorbar()
    plt.show()

_seismogram()

def oppgave2a():
    N = 1000
    t = seismisk_data['t']
    plt.plot(np.linspace(t[0], t[-1], len(h1)), h1)
    plt.plot(np.linspace(t[0], t[-1], len(h2)), h2)
    plt.title('h1(t) og h2(t)')
    plt.xlabel('Tid i sekunder')
    plt.ylabel('Magnitude')
    plt.show()

    f_samp = 1/t[1]-t[0] #f=1/T
    plt.plot(frekspekin3190(h1, N, f_samp)[0], 20*np.log(abs(frekspekin3190(h1, N, f_samp)[1])))
    plt.plot(frekspekin3190(h2, N, f_samp)[0], 20*np.log(abs(frekspekin3190(h2, N, f_samp)[1])))
    plt.title('Frekvensspekterene til h1 og h2')
    plt.xlabel('Frekvens i Hz')
    plt.ylabel('Utslag i dB')
    plt.show()

oppgave2a()

#oppgave2b
def oppgave2b():
    N = 1000
    f_samp = 1/t[1]-t[0]
    tmp = seismogram1[:,75][:400]
    tid=t[:400]

    plt.plot(tid, tmp, label = 'Nærtrase uten vindusfunksjon')
    plt.plot(t[:400], tmp*signal.tukey(len(tmp)), label = 'Nærtrase med vindusfunksjon')
    plt.title('Nærtrasene uten og med vindusfunksjon ')
    plt.xlabel('Tid i sekunder')
    plt.ylabel('Utslag')
    plt.legend()
    plt.show()

    plt.plot(frekspekin3190(tmp, N, f_samp)[0], 20*np.log10(abs(frekspekin3190(tmp, N, f_samp)[1])),
    label = 'Frekvenspekter uten vindusfunksjon')
    plt.plot(frekspekin3190(tmp, N, f_samp)[0], 20*np.log10(abs(frekspekin3190(tmp*signal.tukey(len(tmp)), N, f_samp)[1])),
    label = 'Frekvenspekter med vindusfunksjon')
    plt.title('Frekvensspekterene uten og med vindusfunksjon i decibel')
    plt.xlabel('Frekvens')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()

oppgave2b()

#oppgave2c

def oppgave2c():
    plt.figure(1)
    plt.imshow(seismogram1[0:500, 0:300], cmap='gray', aspect='auto',
    extent=[offset1[0], offset1[300], t[500], t[0]])
    plt.colorbar()
    plt.title('Seismisk gather 1.\nIkke filtrert')
    plt.xlabel('Offset i meter')
    plt.ylabel('Tid i sekunder')


    fseismogram = np.zeros(seismogram1.shape)
    for i in range(fseismogram.shape[1]):
        fseismogram[:,i] = konvin3190(h1, seismogram1[:,i], 0)
    plt.figure(2)
    plt.imshow(fseismogram[0:500, 0:300], cmap='gray', aspect='auto',
    extent=[offset1[0], offset1[300], t[500], t[0]])
    plt.colorbar()
    plt.title('Seismisk gather 1.\n Filtrert med filter h1')
    plt.xlabel('Offset i meter')
    plt.ylabel('Tid i sekunder')
    plt.show()

oppgave2c()

#oppgave3a+b
def oppgave3():
    fseismogram = np.zeros(seismogram1.shape)
    for i in range(fseismogram.shape[1]):
        fseismogram[:,i] = konvin3190(h1, seismogram1[:,i], 0)
    plt.imshow(fseismogram[40:200,0:100], cmap='gray', aspect='auto',
    extent = [offset1[0], offset1[100], t[200], t[40]])
    plt.colorbar()
    plt.title('Seismisk gather ved direkte ankomst')
    plt.xlabel('Offset i meter')
    plt.ylabel('Tid i sekunder')
    plt.show()

    N = 1000

    f_samp = 1/(t[1]-t[0])
    tmp = fseismogram[:,50][:500]
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(t[:500], tmp*signal.tukey(len(tmp)))
    plt.title('''Signalet ved offset {} meter,
    med anvendt filter og tukey vindusfunksjon.'''.format(offset1[50]))
    plt.xlabel('Tid i sekunder')
    plt.ylabel('Utslag')

    plt.subplot(1, 2, 2)
    f_tmp = frekspekin3190(tmp*signal.tukey(len(tmp)),N,f_samp)
    plt.plot(f_tmp[0], 20*np.log(abs(f_tmp[1])), label='Frekvenspekter')
    plt.plot(abs(f_tmp[0][abs(f_tmp[1]).argmax()]), 20*np.log(abs(f_tmp[1]).max()), 'ro',
    label = 'Dominerende frekvens ; {0:.2f} Hz'.format(abs(f_tmp[0][abs(f_tmp[1]).argmax()])))
    plt.legend()
    plt.title('''Frekvenspekter ved offset {} meter,
    med anvendt filter og tukey vindusfunksjon.'''.format(offset1[50]))
    plt.xlabel('Frekvens i Hz')
    plt.ylabel('Magnitude i dB')
    plt.show()
oppgave3()

def oppgave4():
    fseismogram = np.zeros(seismogram1.shape)
    for i in range(fseismogram.shape[1]):
        fseismogram[:,i] = konvin3190(h1, seismogram1[:,i], 0)
    plt.imshow(fseismogram[100:1500, 0:600], cmap='gray', aspect = 'auto',
    extent = [offset1[0], offset1[600], t[1500], t[100] ], vmin = -0.00070,
    vmax = 0.000700)
    plt.colorbar()

    plt.scatter(offset1[0], t[225], color = 'b', marker = 'x')

    primær_mutipel_tid = np.array([410, 595, 795, 995])
    offset_1 = np.zeros(primær_mutipel_tid.size) + offset1[0]

    plt.scatter(offset_1, t[primær_mutipel_tid], color = 'b', marker = 'o')

    plt.scatter(offset1[0], t[615], color = 'r', marker = 'x')

    primær_mutipel_tid = np.array([805, 1005])
    offset_1 = np.zeros(primær_mutipel_tid.size) + offset1[0]

    plt.scatter(offset_1, t[primær_mutipel_tid], color = 'r', marker = 'o')

    plt.title('Seismisk Gather 1.')
    plt.xlabel('Offset i meter')
    plt.ylabel('Tid i sekunder')
    plt.show()
oppgave4()


def oppgave5():
    fseismogram = np.zeros(seismogram1.shape)
    for i in range(fseismogram.shape[1]):
        fseismogram[:,i] = konvin3190(h1, seismogram1[:,i], 0)
    plt.imshow(fseismogram[40:200, 0:100], cmap='gray', aspect = 'auto',
    extent = [offset1[0], offset1[100], t[200], t[40] ])
    plt.colorbar()
    plt.title('Seismisk gather 1 ved direkte ankomst')
    plt.xlabel('Offset i meter')
    plt.ylabel('Tid i sekunder')
    plt.show()

    N = 1000
    _n = 200
    _k = 0
    d_off = 60
    f_samp = 1/t[1]-t[0]
    tmp_1 = abs(fseismogram[:,(_k)][:_n])
    tmp_2 = abs(fseismogram[:,(_k+d_off)][:_n])
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(t[:_n], tmp_1)

    _max1 = [t[:_n][tmp_1.argmax()][0], tmp_1.max()]
    plt.plot(_max1[0], _max1[1], 'ro', label = 'Max ; {} Sekunder'.format(_max1[0]))
    plt.legend()
    plt.title('''Signalet ved offset {} meter ved direkter ankomst'''.format(offset1[_k]))
    plt.xlabel('Tid i sekunder')
    plt.ylabel('Utslag')

    plt.subplot(1, 2, 2)
    plt.plot(t[:_n], tmp_2)
    _max2 = [t[:_n][tmp_2.argmax()][0], tmp_2.max()]
    plt.plot(_max2[0], _max2[1], 'ro', label = 'Max ; {} Sekunder'.format(_max2[0]))
    plt.legend()
    plt.title('''Signalet ved offset {} meter ved direkter ankomst'''.format(offset1[_k+d_off]))
    plt.xlabel('Tid i sekunder')
    plt.ylabel('Utslag')
    plt.show()
    print('Hastigheten til lyden i vannlget er {0:.02f} m/s'
    .format((offset1[_k+d_off] - offset1[_k]) / (_max2[0]-_max1[0])))
oppgave5()


#Jeg bruker L.Uieda sin NMO korreksjonsfunksjon til å rette ut refleksjonene i skuddgatherene.
def nmo_correction(cmp, dt, offsets, velocities):
    nmo = np.zeros_like(cmp)
    nsamples = cmp.shape[0]
    times = np.arange(0, nsamples*dt, dt)
    for i, t0 in enumerate(times):
        for j, x in enumerate(offsets):
            t = reflection_time(t0, x, velocities[i])
            amplitude = sample_trace(cmp[:, j], t, dt)
            if amplitude is not None:
                nmo[i, j] = amplitude
    return nmo

def reflection_time(t0, x, vnmos):
    t = np.sqrt(t0**2 + x**2/vnmos**2)
    return t

from scipy.interpolate import CubicSpline

def sample_trace(trace, time, dt):
    before = int(np.floor(time/dt))
    N = trace.size
    samples = np.arange(before - 1, before + 3)
    if any(samples < 0) or any(samples >= N):
        amplitude = None
    else:
        times = dt*samples
        amps = trace[samples]
        interpolator = CubicSpline(times, amps)
        amplitude = interpolator(time)
    return amplitude

def oppgave6():
    fseismogram = np.zeros(seismogram1.shape)
    for i in range(fseismogram.shape[1]):
        fseismogram[:,i] = konvin3190(h1, seismogram1[:,i], 0)

    velocities = np.zeros(t.size) + 1453  # what??
    _tmp = nmo_correction(fseismogram, t[1]-t[0], offset1, velocities)

    plt.imshow(_tmp, cmap ='gray', aspect = 'auto',
    extent = [offset1[0], offset1[-1], t[-1], t[0]], vmin = -0.00070, vmax = 0.00070)
    plt.colorbar()
    plt.title('Det NMO-korrigerte gatheret med konstante hastigheter {} m/s'.format(velocities[0]))
    plt.xlabel('Offset i meter')
    plt.ylabel('Tid i sekunder')
    plt.show()

    velocities = np.zeros(t.size) + 2450
    _tmp = nmo_correction(fseismogram, t[1]-t[0], offset1, velocities)
    plt.imshow(_tmp, cmap = 'gray', aspect = 'auto',
    extent = [offset1[0], offset1[-1], t[-1], t[0]], vmin = -0.00070, vmax = 0.00070)
    plt.colorbar()
    plt.title('Det NMO-korrigerte gatheret med konstante hastigheter {} m/s'.format(velocities[0]))
    plt.xlabel('Offset i meter')
    plt.ylabel('Tid i sekunder')
    plt.show()


def oppgave7a():
    fseismogram = np.zeros(seismogram1.shape)
    for i in range(fseismogram.shape[1]):
        fseismogram[:,i] = konvin3190(h1, seismogram1[:,i], 0)
    plt.imshow(fseismogram[600:800, 500:600], cmap='gray', aspect = 'auto',
    extent = [offset1[500], offset1[600], t[800], t[600]],
    vmin = -0.00001, vmax = 0.00001 )
    plt.colorbar()
    plt.title('Seismisk Gather 1.\n Første refraksjon ved fjerntrasene')
    plt.xlabel('Offset i Meter')
    plt.ylabel('Tid i sekunder')
    plt.show()

    N = 1000
    _t0 = 600
    _tn = 800
    _k = 490
    d_off = 100
    f_samp = 1/t[1]-t[0]
    tmp_1 = abs(fseismogram[:,(_k)][_t0:_tn])
    tmp_2 = abs(fseismogram[:,(_k+d_off)][_t0:_tn])

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(t[_t0:_tn], tmp_1)

    _max1 = [t[_t0:_tn][tmp_1.argmax()][0], tmp_1.max()]

    plt.plot(_max1[0], _max1[1], 'ro', label = 'Max ; {} Sekunder'.format(_max1[0]))
    plt.legend()
    plt.xlabel('Tid i sekunder')
    plt.ylabel('Utslag')
    plt.title('''Signalet ved offset {} meter ved refraksjonen som returnerer\n
    fra grensesjiktet mellom vannlaget og 1. sedimentærlaget'''
    .format(offset1[_k]), size=8)

    plt.subplot(1, 2, 2)
    plt.plot(t[_t0:_tn], tmp_2)
    _max2 = [t[_t0:_tn][tmp_2.argmax()][0], tmp_2.max()]
    plt.plot(_max2[0], _max2[1], 'ro', label = 'Max ; {} Sekunder'.format(_max2[0]))
    plt.legend()
    plt.xlabel('Tid i sekunder')
    plt.ylabel('Utslag')
    plt.title('''Signalet ved offset {} meter ved refraksjonen som returnerer\n
    fra grensesjiktet mellom vannlaget og 1. sedimentærlaget'''
    .format(offset1[_k+d_off]), size=8)
    plt.show()
    print('Hastigheten til lyden i det første sedimentærlaget er {0:.02f} m/s'
    .format(( (offset1[_k+d_off] - offset1[_k]) / (_max2[0] - _max1[0]) )))
oppgave7a()


def oppgave7b():
    fseismogram = np.zeros(seismogram1.shape)
    for i in range(fseismogram.shape[1]):
        fseismogram[:,i] = konvin3190(h1, seismogram1[:,i], 0)
    plt.imshow(fseismogram[800:1000, 500:600], cmap='gray', aspect = 'auto',
    extent = [offset1[500], offset1[600], t[1000], t[800]],
    vmin = -0.00001, vmax = 0.00001 )
    plt.colorbar()
    plt.title('''Seismisk gather 1\n
    Første refraksjon fra andre laget ved fjerntrasene''')
    plt.xlabel('Offset i meter')
    plt.ylabel('Tid i sekunder')
    plt.show()

    N = 1000
    _t0 = 800
    _tn = 1000
    _k = 560
    d_off = 30
    f_samp = 1/t[1]-t[0]
    tmp_1 = abs(fseismogram[:,(_k)][_t0:_tn])
    tmp_2 = abs(fseismogram[:,(_k+d_off)][_t0:_tn])

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(t[_t0:_tn], tmp_1)
    _max1 = [t[_t0:_tn][tmp_1.argmax()][0], tmp_1.max()]
    plt.plot(_max1[0], _max1[1], 'ro', label = 'Max ; {} Sekunder'.format(_max1[0]))
    plt.legend()
    plt.xlabel('Tid i sekunder')
    plt.ylabel('Utslag')
    plt.title('''Signalet ved offset {} meter ved refraksjonen som returnerer\n
    fra grensesjiktet mellom vannlaget og det dypeste sedimentærlaget'''
    .format(offset1[_k]), size=8)

    plt.subplot(1, 2, 2)
    plt.plot(t[_t0:_tn], tmp_2)
    _max2 = [t[_t0:_tn][tmp_2.argmax()][0], tmp_2.max()]
    plt.plot(_max2[0], _max2[1], 'ro', label = 'Max ; {} Sekunder'.format(_max2[0]))
    plt.legend()
    plt.xlabel('Tid i sekunder')
    plt.ylabel('Utslag')
    plt.title('''Signalet ved offset {} meter ved refraksjonen som returnerer\n
    fra grensesjiktet mellom vannlaget og det dypeste sedimentærlaget'''
    .format(offset1[_k+d_off]), size=8)
    plt.show()
    print('Hastigheten til lyden i det andre sedimentærlaget er {0:.02f} m/s'
    .format(( (offset2[_k+d_off] - offset2[_k]) / (_max2[0] - _max1[0]) )))

oppgave7b()


def oppgave7c():
    fseismogram = np.zeros(seismogram2.shape)
    for i in range(fseismogram.shape[1]):
        fseismogram[:,i] = konvin3190(h1, seismogram2[:,i], 0)
    plt.imshow(fseismogram[900:1100, 600:800], cmap='gray', aspect = 'auto',
    extent = [offset2[600], offset2[800], t[1100], t[900]],
    vmin = -0.00001, vmax = 0.00001 )
    plt.colorbar()
    plt.title('''Seismisk gather 2\n
    Første refraksjon fra andre laget ved fjerntrasene''')
    plt.xlabel('Offset i meter')
    plt.ylabel('Tid i sekunder')
    plt.show()

    N = 1000
    _t0 = 900
    _tn = 1100
    _k = 760
    d_off = 20
    f_samp = 1/t[1]-t[0]
    tmp_1 = abs(fseismogram[:,(_k)][_t0:_tn])
    tmp_2 = abs(fseismogram[:,(_k+d_off)][_t0:_tn])

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(t[_t0:_tn], tmp_1)
    _max1 = [t[_t0:_tn][tmp_1.argmax()][0], tmp_1.max()]
    plt.plot(_max1[0], _max1[1], 'ro', label = 'Max ; {} Sekunder'.format(_max1[0]))
    plt.legend()
    plt.xlabel('Tid i sekunder')
    plt.ylabel('Utslag')
    plt.title('''Signalet ved offset {} meter ved refraksjonen som returnerer\n
    fra grensesjiktet mellom vannlaget og det dypeste sedimentærlaget'''
    .format(offset2[_k]), size=8)

    plt.subplot(1, 2, 2)
    plt.plot(t[_t0:_tn], tmp_2)
    _max2 = [t[_t0:_tn][tmp_2.argmax()][0], tmp_2.max()]
    plt.plot(_max2[0], _max2[1], 'ro', label = 'Max ; {} Sekunder'.format(_max2[0]))
    plt.legend()
    plt.xlabel('Tid i sekunder')
    plt.ylabel('Utslag')
    plt.title('''Signalet ved offset {} meter ved refraksjonen som returnerer\n
    fra grensesjiktet mellom vannlaget og det dypeste sedimentærlaget'''
    .format(offset2[_k+d_off]), size=8)
    plt.show()

    print('Hastigheten til lyden i det andre sedimentærlaget er {0:.02f} m/s'
    .format(( (offset2[_k+d_off] - offset2[_k]) / (_max2[0] - _max1[0]) )))

oppgave7c()


v0 = 1456.31
v1 = 2631.58
v2 = 3125.00

t0 = t[225][0]
t1 = t[615][0]

d0 = np.sqrt((offset1[0])**2 + (v0*t0/2)**2)
d1 = np.sqrt( ((offset1[0])**2) + (( ( (v0*t0) + (v1*(t1-t0)))/2)**2 )) -d0

print('Dybden i havet er {0:.02f} meter'.format(d0))
print('Dybden til det 1. sedimentærlaget er tilnærmet {0:.02f} meter'.format(d1))

