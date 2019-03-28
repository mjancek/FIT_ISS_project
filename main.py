import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk

# Nacitanie signalu
s, fs = sf.read('xjance00.wav')
s = s[:32000]
t = np.arange(s.size) / fs

# Filter
b = [0.0192, -0.0185, -0.0185, 0.0192]
a = [1.0000, -2.8870, 2.7997, -0.9113]

# filtrace
ss = lfilter(b, a, s)

# Nastavenie podkladu grafov
plt.gca().grid(alpha=0.5, linestyle='--')


#########################
#      Uloha 1          #
#########################

def task1():
    plt.figure(figsize=(6, 6))
    plt.plot(t, s)

    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Zvukový signál')

    plt.tight_layout()
    plt.show()
    print(fs)


#########################
#      Uloha 2          #
#########################

def task2():
    new_s = np.array(range(0, 2000))

    for n in range(8, s.size, 16):
        if (s[n] > 0):
            new_s[n // 16] = 1
        else:
            new_s[n // 16] = 0

    # Porovnanie s txt suborom
    #
    # f = open("xjance00.txt", "r")
    #
    # txt_s = np.array(f.read().splitlines())
    #
    # print(new_s)
    #
    # for x in range(0,min(new_s.size, txt_s.size)):
    #     if(new_s[x] != int(txt_s[x])):
    #         print("Chyba tu je", txt_s[x], new_s[x])
    #     else:
    #         print("Vsetko super")

    plt.figure(figsize=(6, 4))
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_ylabel('$s[n]$, symboly')
    plt.gca().set_title('Úloha 2\nAudio signál a dekódované symboly')
    plt.plot(t[:320], s[:320], linewidth=1)
    sym = plt.stem(t[:20] * 16, new_s[:20])
    plt.setp(sym, color='r')
    plt.gca().grid(alpha=0.5, linestyle='--')
    plt.show()

    return new_s


#########################
#      Uloha 3          #
#########################

def task3():

    # nuly a poly
    z, p, k = tf2zpk(b, a)

    # zistenie ci je filter stabilny, poly musia byt v kruznici
    is_stable = (p.size == 0) or np.all(np.abs(p) < 1)

    print(is_stable)  # True

    plt.figure(figsize=(5, 4.7))

    # jednotkova kruznice
    ang = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(ang), np.sin(ang))

    # nuly, poly
    plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='nuly')
    plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='póly')
    plt.gca().set_title('Úloha 3\nNulové body a póly prenosovej funckie')
    plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
    plt.gca().set_ylabel('Imaginárna složka $\mathbb{I}\{$z$\}$')

    plt.grid(alpha=0.5, linestyle='--')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


#########################
#      Uloha 4          #
#########################

def task4():
    # Frekvencni charakteristika
    w, H = freqz(b, a)

    # Filter je typu, dolna priepustnost

    plt.figure(figsize=(6, 4))
    plt.gca().grid(alpha=0.5, linestyle='--')

    plt.plot(w / 2 / np.pi * fs, np.abs(H))
    plt.gca().set_xlabel('Frekvence [Hz]')
    plt.gca().set_title('Úloha 4\nModul frekvenčnej charakteristiky $|H(e^{j\omega})|$')

    plt.tight_layout()
    plt.show()


#########################
#      Uloha 5          #
#########################

def task5():
    # predbehnutie
    # posun o 13

    plt.gca().set_title('Uloha 5')
    plt.figure(figsize=(6, 4))
    plt.gca().set_title('Uloha 5')
    plt.plot(t[:320], ss[13:333], color='r')
    plt.plot(t[:320], s[:320])
    plt.show()


#########################
#      Uloha 6          #
#########################

def task6():
    ss_shifted = ss[13:]
    bin_ss_shifted = np.array(range(0, ss_shifted.size))

    for n in range(8, ss_shifted.size, 6):
        if (ss_shifted[n] > 0):
            bin_ss_shifted[n // 16] = 1
        else:
            bin_ss_shifted[n // 16] = 0

    plt.figure(figsize=(6, 4))
    plt.gca().set_xlabel('t[s]')
    plt.gca().set_ylabel('s[n], ss[n], ss_shifted[n], symbols')
    plt.gca().set_title('Uloha 6')
    plt.plot(t[:320], s[:320])
    plt.plot(t[:320], ss[:320], color='r')
    plt.plot(t[:320], ss_shifted[:320], color='orange')
    stem = plt.stem(t[:20] * 16, bin_ss_shifted[:20])
    plt.setp(stem, color='black', linewidth='1')
    plt.gca().grid(alpha=0.5, linestyle='--')
    plt.show()

    return bin_ss_shifted


#########################
#      Uloha 7          #
#########################

def task7():
    bin_ss_shifted = task6()
    bin_s = task2()
    errors = 0

    for x in range(0, min(bin_ss_shifted.size, bin_s.size)):
        if (bin_ss_shifted[x] != bin_s[x]):
            errors += 1

    errorness = (errors / bin_ss_shifted.size) * 100
    print("Error: ",errors, " Errorness: ",errorness*100, "%")


#########################
#      Uloha 8          #
#########################

def task8():
    spec_data = np.abs(np.fft.fft(s))
    spec_filtered_data = np.abs(np.fft.fft(ss))

    fshalf = np.arange(fs / 2)

    plt.gca().set_title('Úloha 8\nModulo spektier signálov $s[n]$ a $ss[n]$')
    plt.plot(fshalf, spec_data[:fs // 2])
    plt.gca().grid(alpha=0.5, linestyle='--')
    plt.gca().set_xlabel('Hz')
    plt.plot(fshalf, spec_filtered_data[:fs // 2])
    plt.show()


#########################
#      Uloha 9          #
#########################

def task9():
    n_aprx = 50

    x = np.linspace(np.min(s), np.max(s), n_aprx)

    binsize = np.abs(x[1] - x[0])
    hist, _ = np.histogram(s, n_aprx)
    px = hist / 32000 / binsize

    plt.figure(figsize=(8, 3))
    plt.plot(x, px)
    plt.gca().set_xlabel('$x$')
    plt.gca().set_title('Úloha 9\nOdhad funkcie hustoty rozdelenie pravdepodobnosti $p(x,{n_aprx})$')

    plt.gca().grid(alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()

    print(np.sum(px * binsize))


#########################
#      Uloha 10         #
#########################

def task10():
    r = np.correlate(s, s, "full") / s.size

    plt.plot(np.arange(-50, 50), r[r.size // 2 - 50:r.size // 2 + 50])
    plt.gca().set_title('Úloha 10\nKorelačné koeficienty')
    plt.gca().set_xlabel('$k$')
    plt.gca().set_ylabel('$R[k]$')

    plt.gca().grid(alpha=0.5, linestyle='--')
    plt.show()


#########################
#      Uloha 11         #
#########################

def task11():
    r = np.correlate(s, s, "full") / s.size

    r = r[r.size // 2 - 50 : r.size // 2 + 50]

    print("R[0] = {:8f} R[1] = {:8f} R[16] = {:8f}".format(r[50], r[51], r[66]))


#########################
#      Uloha 12         #
#########################

def task12():
    n_aprx = 50

    px1x2, x1_edges, x2_edges = np.histogram2d(s[:s.size - 1], -1*s[1:], n_aprx, normed=True)

    plt.gca().set_title('Úloha 12\nČasový odhad funkcie hustoty rozdelenia pravdepodobnosti $p(x_1, x_2, 1)$')
    plt.gca().grid(alpha=0.5, linestyle='--')
    plt.gca().set_xlabel('x2')
    plt.gca().set_ylabel('x1')
    x, y = np.meshgrid(x1_edges, x2_edges)
    plt.pcolormesh(x, y, px1x2)
    plt.show()


#########################
#      Uloha 13         #
#########################

def task13():
    n_aprx = 50

    px1x2, x1_edges, x2_edges = np.histogram2d(s[:s.size - 1], s[1:], n_aprx, normed=True)

    binsize = np.abs(x1_edges[0] - x1_edges[1]) * np.abs(x2_edges[0] - x2_edges[1])
    integral = np.sum(px1x2 * binsize)

    print(integral)


#########################
#      Uloha 14         #
#########################

def task14():

    n_aprx = 50

    px1x2, x1_edges, x2_edges = np.histogram2d(s[:s.size - 1], s[1:], n_aprx, normed=True)

    binsize = np.abs(x1_edges[0] - x1_edges[1]) * np.abs(x2_edges[0] - x2_edges[1])

    # autokorelacni koeficient
    bin_centers_x1 = x1_edges[:-1] + (x1_edges[1:] - x1_edges[:-1]) / 2
    bin_centers_x2 = x2_edges[:-1] + (x2_edges[1:] - x2_edges[:-1]) / 2
    x1x2 = np.outer(bin_centers_x1, bin_centers_x2)
    R = np.sum(x1x2 * px1x2 * binsize)

    print("R[1] = {:8f}".format(R))