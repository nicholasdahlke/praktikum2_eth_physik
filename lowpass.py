from numpy.ma.core import sqrt
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
data = "data_praktikum_2/Tiefpass"

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
    ch1_data = np.loadtxt(ch1, delimiter=",", usecols=range(3,5)) # U_in
    ch2_data = np.loadtxt(ch2, delimiter=",", usecols=range(3,5)) # U_out
    ch1_data[:,0] -= np.min(ch1_data[:,0])
    ch2_data[:,0] -= np.min(ch2_data[:,0])

    """resamp_num = int(1e6)
    ch1_data = np.zeros((resamp_num, 2))
    ch2_data = np.zeros((resamp_num, 2))
    ch1_data[:,0] = np.linspace(np.min(ch1_data[:,0]), np.max(ch1_data[:,0]), resamp_num)
    ch2_data[:,0] = np.linspace(np.min(ch2_data[:,0]), np.max(ch2_data[:,0]), resamp_num)

    ch1_data[:,1] = np.interp(ch1_data[:,0], ch1_data[:,0], ch1_data[:,1])
    ch2_data[:,1] = np.interp(ch1_data[:,0], ch2_data[:,0], ch2_data[:,1])"""

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

    ratio = 10*np.log(np.mean(np.pow(ch2_data[:,1],2),)/np.mean(np.pow(ch1_data[:,1],2),))
    print(f"Ratio: {ratio}")
    print(50*"*")

    # Phase berechnen
    fft_ch2 = np.fft.fft(ch2_data[:, 1])
    phase_shift = np.angle(fft_ch2[idx]) - np.angle(fft[idx])

    # Strom über Kondensator
    u_c = ch1_data[:,1] - ch2_data[:,1]
    i_c = u_c / r
    fft_ic = np.fft.fft(u_c)
    phase_shift_ic = np.angle(fft_ic[idx]) - np.angle(fft_ch2[idx])
    phase_shift_ic = np.where(phase_shift_ic < 0, phase_shift_ic + 2*np.pi, phase_shift_ic)
    results.append([frequency, ratio, phase_shift, phase_shift_ic])


results = np.array(results)
results[:,0] *= 2*np.pi # Umrechnung in rad/s
figs = (6.95, 3)
fig, ax = plt.subplots(layout="constrained", figsize=figs)
ax.scatter(results[:,0], results[:,1], label="$Q$ Measurements", color="tab:blue")

transfer_function = np.vectorize(lambda f, c, r: 10*np.log((1/sqrt(((f)**2 * r**2 * c**2) + 1))**2))
f = np.linspace(0,1e5, 1000)*2*np.pi
ax.plot(f, transfer_function(f, c, r), label="$Q$ Theory", color='tab:blue')

#results[:,1] = np.pow(results[:,1])
popt, pcov = curve_fit(transfer_function, results[:,0], results[:,1], p0=[r, c])
R_fit, C_fit = popt
print(f"Gefittete Werte: R = {R_fit:.2f} Ohm, C = {C_fit*1e9:.2f} nF")
fitted_ratio = transfer_function(f, R_fit, C_fit)
ax.plot(f, fitted_ratio, label="$Q$ Fit", linestyle='dashed', color='tab:blue')

# Grenzfrequenz finden, wo fitted_ratio = -3 dB
idx = np.argmin(np.abs(fitted_ratio - (-6)))
cutoff_f = f[idx]

#plt.legend()

ax.set_xscale('log')
ax.xaxis.set_minor_locator(ticker.LogLocator(subs='all'))
ax.grid(True, which='both')
ax.set_ylabel(r"$Q$ [dB]")
ax.set_xlabel(r"$\omega$ [rad/s]")


# Zweite y-Achse für results[:,2]
ax2 = ax.twinx()
phase = np.vectorize(lambda f, r, c: -np.arctan(f*r*c))
ax2.scatter(results[:,0], results[:,2], label=r"$\varphi$ Measurement", color="tab:orange")
ax2.plot(f, phase(f, r, c), color='tab:orange', label=r"$\varphi$ Theory")
ax2.set_ylabel(r"$\varphi$ [rad]")
ax2.tick_params(axis='y')
#ax2.scatter(results[:,0],-results[:,3],marker="x", color="green")


# Achse in Einheiten von π
space = 0.05
ax2_bottom = 0
ax2_top = -np.pi
ax2_dist = abs(ax2_bottom - ax2_top)
ax2.set_ylim(ax2_bottom+ax2_dist*space, ax2_top - ax2_dist*space)
ax2.set_yticks([0, -np.pi/4, -np.pi/2, -3*np.pi/4, -np.pi])
ax2.set_yticklabels([r'$0$', r'$-\pi/4$', r'$-\pi/2$', r'$-3\pi/4$', r'$-\pi$'])

ax1_bottom = 0
ax1_top = -80
ax1_dist = abs(ax1_bottom - ax1_top)
ax.set_ylim(ax1_top - ax1_dist*space, ax1_bottom+ax1_dist*space)
ax.set_yticks([0, -20, -40, -60, -80])
ax.set_yticklabels([r'$0$', r'$-20$', r'$-40$', r'$-60$', r'$-80$'])



# Horizontale Linie bei π/4
#ax2.axhline(y=-np.pi/4, color='blue', linestyle='--', label=r'$\pi/4$')

# Fitting der Phase-Funktion
print("Phase values:", results[:,2])
print("Phase ic values:", results[:,3])
mask = results[:,2] < 0
print(f"Using {np.sum(mask)} out of {len(mask)} points for phase fit")
phase_func = np.vectorize(lambda f, r, c: -np.arctan(f*r*c))
popt_phase, pcov_phase = curve_fit(phase_func, results[mask,0], results[mask,2], p0=[R_fit, C_fit], bounds=([0, 0], [1e6, 1e-3]))
R_phase, C_phase = popt_phase
print(f"Gefittete Werte Phase: R = {R_phase:.2f} Ohm, C = {C_phase*1e9:.2f} nF")
fitted_phase = phase_func(f, R_phase, C_phase)
ax2.plot(f, fitted_phase, linestyle='dashed', color='tab:orange', label=r"$\varphi$ Fit")
#ax2.legend(loc='lower left')
ax.axvline(x=cutoff_f, color='grey', linestyle='--')

# Combine legends
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
#ax.legend( loc='best')

ax.legend(handles1 + handles2, labels1 + labels2,bbox_to_anchor=(0., 1.05, 1., 0.102), loc="lower left", mode="expand", ncol=3, borderaxespad=0.)
fig.savefig("tiefpass.pdf", bbox_inches='tight')

figs = (3.4, 3)
fig1, ax_new = plt.subplots(layout="constrained", figsize=figs)
ax_new.scatter(results[:, 0], results[:, 3], label=r"$\varphi_\text{C}$ Measurement", color="tab:orange")
ax_new.set_ylim((0, np.pi))
ax_new.set_xscale("log")
ax_new.xaxis.set_minor_locator(ticker.LogLocator(subs='all'))
ax_new.grid(True, which='both')
ax_new.set_ylabel(r"$Q$ [dB]")
ax_new.set_xlabel(r"$\varphi_\text{C}$ [rad]")
ax_new.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
ax_new.set_yticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
ax_new.plot([np.min(results[:,0]), np.max(results[:,0])], [np.pi/2, np.pi/2],  label=r"$\varphi_\text{C}$ Theory", color="tab:orange")
ax_new.legend()

plt.savefig("phase_shift_c.pdf", bbox_inches='tight')
plt.show()

