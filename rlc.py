from numpy.ma.core import sqrt
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
data = "data_praktikum_2/Parallelschwingkreis"

plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 11,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{siunitx}",
})

csv_list = []
r = 1e3
c = 100e-9
l = 10e-3

# Durch alle Unterordner gehen
for folder in os.listdir(data):
    folder_path = os.path.join(data, folder)
    if os.path.isdir(folder_path):
        # CSV-Dateien im Ordner finden
        ch1_file = None
        ch2_file = None

        for file in os.listdir(folder_path):
            if file.endswith("CH1.CSV"):
                ch1_file = os.path.join(folder_path, file)
            elif file.endswith("CH2.CSV"):
                ch2_file = os.path.join(folder_path, file)

        # Wenn beide Dateien gefunden, als Tuple speichern
        if ch1_file and ch2_file:
            csv_list.append((ch1_file, ch2_file))

#csv_list = [csv_list[0]]
results = []

for measurement in csv_list:
    ch1, ch2 = measurement
    print(ch1, ";", ch2)
    ch1_data = np.loadtxt(ch1, delimiter=",", usecols=range(3, 5))  # U_in
    ch2_data = np.loadtxt(ch2, delimiter=",", usecols=range(3, 5))  # U_out
    ch1_data[:, 0] -= np.min(ch1_data[:, 0])
    ch2_data[:, 0] -= np.min(ch2_data[:, 0])

    #plt.plot(ch1_data[:,0], ch1_data[:,1], label="CH1")
    #plt.plot(ch2_data[:,0], ch2_data[:,1], label="CH2")
    #plt.show()

    fft = np.fft.fft(ch1_data[:, 1])
    freqs = np.fft.fftfreq(len(ch1_data[:, 1]), ch1_data[1, 0] - ch1_data[0, 0])
    idx = np.argmax(np.abs(fft[1:len(fft) // 2])) + 1

    analytic = hilbert(ch1_data[:, 1])
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2 * np.pi * (ch1_data[1, 0] - ch1_data[0, 0]))
    frequency = np.mean(inst_freq)
    #frequency = freqs[idx]
    print(f"Frequenz: {frequency} Hz")

    ratio = 10 * np.log(np.mean(np.pow(ch2_data[:, 1], 2), ) / np.mean(np.pow(ch1_data[:, 1], 2), ))
    print(f"Ratio: {ratio}")
    print(50 * "*")

    # Phase berechnen
    fft_ch2 = np.fft.fft(ch2_data[:, 1])
    phase_shift = np.angle(fft_ch2[idx]) - np.angle(fft[idx])

    # Strom über Kondensator
    u_c = ch1_data[:, 1] - ch2_data[:, 1]
    i_c = u_c / r
    fft_ic = np.fft.fft(u_c)
    phase_shift_ic = np.angle(fft_ic[idx]) - np.angle(fft_ch2[idx])
    phase_shift_ic = np.where(phase_shift_ic < 0, phase_shift_ic + 2 * np.pi, phase_shift_ic)
    results.append([frequency, ratio, phase_shift, phase_shift_ic])

results = np.array(results)
results[:, 0] *= 2 * np.pi  # Umrechnung in rad/s
figs = (3.4, 3)
fig, ax = plt.subplots(layout="constrained", figsize=figs)

ax.scatter(results[:, 0], results[:, 1], label="$Q$ Measurements", color="tab:blue")
#ax.set_xscale('log')

resonance = np.vectorize(
    lambda w, c, l, r: -10 * np.log10(1+np.pow(r, 2)*(c/l)*(w*np.sqrt(l*c) - 1/(w*np.sqrt(l*c)))**2))

w = np.linspace(1e3, 1e5, 1000)
ax.plot(w, resonance(w, c, l, r), label="$Q$ Theory", color="tab:blue")

# Fit resonance function to measurements
def resonance_func(w, c, l, r):
    return -10 * np.log10(1 + np.pow(r, 2) * (c/l) * (w*np.sqrt(l*c) - 1/(w*np.sqrt(l*c)))**2)

try:
    popt, pcov = curve_fit(resonance_func, results[:, 0], results[:, 1], p0=[c, l, r], maxfev=10000)
    c_fit, l_fit, r_fit = popt
    print(f"Fitted parameters: c={c_fit:.2e}, l={l_fit:.2e}, r={r_fit:.2e}")

    # Plot fitted resonance curve
    ax.plot(w, resonance_func(w, c_fit, l_fit, r_fit), label="$Q$ Fit", color="tab:blue", linestyle="--")
except Exception as e:
    print(f"Fitting failed: {e}")

space = 0.05
ax2_bottom = 0
ax2_top = -60
ax2_dist = abs(ax2_bottom - ax2_top)
#ax.set_ylim(ax2_top - ax2_dist*space, ax2_bottom+ax2_dist*space,)
ax.legend(loc='upper left')
ax.grid()
ax.set_ylabel(r"$Q$ [dB]")
ax.set_xlabel(r"$\omega$ [rad/s]")
ax.set_xscale("log")

ax.xaxis.set_minor_locator(ticker.LogLocator(subs='all'))
ax.grid(True, which='both')
plt.savefig("resonance.pdf", bbox_inches="tight")
plt.show()
