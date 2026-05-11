from numpy.ma.core import sqrt
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
data = r"C:\Users\nicho\Documents\Oszi Physikpraktikum\Tiefpass"

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
    ch1_data = np.loadtxt(ch1, skiprows=18, delimiter=",", usecols=range(3,5)) # U_in
    ch2_data = np.loadtxt(ch2, skiprows=18, delimiter=",", usecols=range(3,5)) # U_out
    ch1_data[:,0] -= np.min(ch1_data[:,0])
    ch2_data[:,0] -= np.min(ch2_data[:,0])

    #plt.plot(ch1_data[:,0], ch1_data[:,1], label="CH1")
    #plt.plot(ch2_data[:,0], ch2_data[:,1], label="CH2")
    #plt.show()

    fft = np.fft.fft(ch1_data[:, 1])
    freqs = np.fft.fftfreq(len(ch1_data[:, 1]), ch1_data[1, 0] - ch1_data[0, 0])

    idx = np.argmax(np.abs(fft[1:len(fft) // 2])) + 1
    frequency = freqs[idx]
    print(f"Frequenz: {frequency} Hz")

    ratio = 10*np.log(np.mean(np.pow(ch2_data[:,1],2),)/np.mean(np.pow(ch1_data[:,1],2),))
    print(f"Ratio: {ratio}")

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
transfer_function = np.vectorize(lambda f, c, r: 10*np.log((1/sqrt(((f)**2 * r**2 * c**2) + 1))**2))
phase = np.vectorize(lambda f, c, r: -np.arctan(f*r*c))
f = np.linspace(0,1e5, 1000)*2*np.pi
plt.plot(f, transfer_function(f, c, r), label="Theory")

#results[:,1] = np.pow(results[:,1])
popt, pcov = curve_fit(transfer_function, results[:,0], results[:,1], p0=[r, c])
R_fit, C_fit = popt
print(f"Gefittete Werte: R = {R_fit:.2f} Ohm, C = {C_fit*1e9:.2f} nF")
fitted_ratio = transfer_function(f, R_fit, C_fit)
plt.plot(f, fitted_ratio, label='Fit', linestyle='dashed', color='blue')

# Grenzfrequenz finden, wo fitted_ratio = -3 dB
idx = np.argmin(np.abs(fitted_ratio - (-6)))
cutoff_f = f[idx]
plt.axvline(x=cutoff_f, color='red', linestyle='--', label=f'Cutoff at {cutoff_f:.0f} rad/s')

plt.legend()

plt.scatter(results[:,0], results[:,1], label="Measurements")
plt.xscale('log')
plt.grid(True)
plt.ylabel(r"$Q$ [dB]")
plt.xlabel(r"$\omega$ [rad/s]")

# Zweite y-Achse für results[:,2]
ax2 = plt.twinx()
ax2.scatter(results[:,0], results[:,2], label='Phase shift')
ax2.plot(f, phase(f, c,r))
ax2.set_ylabel(r"$\varphi$ [rad]")
ax2.tick_params(axis='y')
#ax2.scatter(results[:,0],-results[:,3],marker="x", color="green")


# Achse in Einheiten von π
ax2.set_ylim(0, -np.pi)
ax2.set_yticks([0, -np.pi/4, -np.pi/2, -3*np.pi/4, -np.pi])
ax2.set_yticklabels([r'$0$', r'$-\pi/4$', r'$-\pi/2$', r'$-3\pi/4$', r'$-\pi$'])

# Horizontale Linie bei π/4
ax2.axhline(y=-np.pi/4, color='blue', linestyle='--', label=r'$\pi/4$')

plt.show()