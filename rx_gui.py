import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random
import threading
import time

# imports related to Radio Tx/Rx

# Tx imports
from tx_mimo_npilot import create_data, create_preamble, create_data_frame, init_tx_dict
from system_tx import SysParUL
import json

# Rx imports
import adi
from rx_mimo_plot_npilot import receiver_MIMO
import re

# glob Rx Params
cfo_glob_dummy = 0.001

cfg_test = (2, 'MMSE_rep4') # set config for reception

mimo, rec_name = cfg_test

try:
    pilot_rep_use = int(re.findall('_rep(\d+)', rec_name)[0])
except:
    pilot_rep_use = 1

# some global varaibles related to TxScheme and data
# imitate tranmitted
inPar = init_tx_dict()
(repeated_frame_tx_orig, repeated_frame, frame_len, preamble_len, preamble_core, mod_dict_data, pilot_tx,
 bite_stream_tx, mod_data_tx, bite_stream_tx_uncode)  \
        = create_data_frame(inPar)

# SDR config
if not inPar.dummyRx:
    # additional params
    do_load_file = False # should be False
    do_save = False

    sdr = adi.ad9361(uri='ip:192.168.2.2')
    samp_rate = inPar.sample_rate  # must be <=30.72 MHz if both channels are enabled
    num_samps = len(repeated_frame) * 1  # number of samples per buffer.  Can be different for Rx and Tx
    rx_lo = int(inPar.center_freq)
    rx_mode = "slow_attack"  # can be "manual" or "slow_attack"
    rx_gain0 = 70
    rx_gain1 = 70
    tx_lo = rx_lo
    tx_gain0 = -10
    tx_gain1 = -10

    sdr.rx_enabled_channels = [0, 1]
    #sdr.rx_enabled_channels = [0]
    sdr.sample_rate = int(samp_rate)
    sdr.rx_lo = int(rx_lo)
    sdr.gain_control_mode = rx_mode
    sdr.rx_hardwaregain_chan0 = int(rx_gain0)
    sdr.rx_hardwaregain_chan1 = int(rx_gain1)
    sdr.rx_buffer_size = int(num_samps)


# end of global params



class RxGUI:
    def __init__(self, root):
        self.root = root
        root.title("Real-Time Sensor Data Monitor (Last 20 Points)")

        # Initialize data structures
        self.window_size = 60  # Show last 20 points
        self.time_points = []
        self.data = {
            'SNRguard': [],
            'BER': [],
            'EVM': []
        }

        # Current metric to display
        self.current_metric = 'SNRguard'
        self.counter = 0

        # Create GUI elements
        self.setup_ui()

        # Start sensor simulation thread
        self.running = True
        self.thread = threading.Thread(target=self.sensor_simulator)
        self.thread.daemon = True
        self.thread.start()

        # Start GUI update loop
        self.update_plot()

    def setup_ui(self):
        # Create figure for plotting
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.line, = self.ax.plot([], [], 'b-')
        self.ax.set_title(f"Real-Time {self.current_metric} (Last 20 Points)")
        self.ax.set_xlabel("Most Recent Time Points")
        self.ax.set_ylabel(self.current_metric)
        self.ax.grid(True)

        # Create canvas for matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create radio buttons for metric selection
        self.metric_var = tk.StringVar(value=self.current_metric)

        metrics_frame = ttk.LabelFrame(self.root, text="Select Metric")
        metrics_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        for metric in self.data.keys():
            rb = ttk.Radiobutton(
                metrics_frame,
                text=metric,
                variable=self.metric_var,
                value=metric,
                command=self.metric_changed
            )
            rb.pack(side=tk.LEFT, padx=5, pady=2)

    def metric_changed(self):
        self.current_metric = self.metric_var.get()
        self.ax.set_title(f"Real-Time {self.current_metric} (Last 20 Points)")
        self.ax.set_ylabel(self.current_metric)
        self.update_plot_data()

    def update_plot_data(self):
        if len(self.time_points) > 0:
            # Get the last 20 points or all points if less than 20
            start_idx = max(0, len(self.time_points) - self.window_size)
            x_data = self.time_points[start_idx:]
            y_data = self.data[self.current_metric][start_idx:]

            self.line.set_data(x_data, y_data)

            # Adjust x-axis to always show exactly 20 points (or less if not enough data)
            if len(x_data) >= self.window_size:
                self.ax.set_xlim(x_data[0], x_data[-1])
            else:
                self.ax.set_xlim(0, self.window_size)

            # Auto-scale y-axis based on visible data
            if len(y_data) > 0:
                min_val = min(y_data)
                max_val = max(y_data)
                margin = (max_val - min_val) * 0.1  # 10% margin
                self.ax.set_ylim(min_val - margin, max_val + margin)

            self.canvas.draw()

    def update_plot(self):
        if self.running:
            self.update_plot_data()
            self.root.after(100, self.update_plot)  # Update every 100ms

    def sensor_simulator(self):
        """Simulates sensor data arriving every 0.1 seconds"""
        while self.running:
            # Generate new data point

            # update Rx status
            if inPar.dummyRx:
                # fill blanks
                new_SNR_guard = 3
                new_BER = 0.1
                new_EVM = 1.0

                # dummpy Rx
                repeated_frame_tx = repeated_frame_tx_orig.T
                Nrx_test = 2
                repeated_frame_tx = np.tile(repeated_frame_tx, (Nrx_test, 1))


                SNR = inPar.SNR_dummy
                Es_tmp = np.linalg.norm(repeated_frame_tx.flatten()) ** 2 / len(repeated_frame_tx.flatten())
                sigma_noise_tmp = Es_tmp * 10 ** (-SNR / 10)

                noise_arr = np.sqrt(sigma_noise_tmp / 2) * ((np.random.normal(0, 1, repeated_frame_tx.shape)) + 1j * (
                    np.random.normal(0, 1, repeated_frame_tx.shape)))

                data_rx_dummy = repeated_frame_tx + noise_arr
                SNR_est = 10.0 * np.log10(
                    np.linalg.norm(repeated_frame_tx.flatten()) ** 2 / np.linalg.norm(noise_arr.flatten()) ** 2)
                # print(f'dummy Tx SNRset={SNR:.2f} SNR={SNR_est:.2f}')

                # add dummpy CFO
                cfo_set = cfo_glob_dummy

                # random phase for both channels
                a_amp = np.random.rand(repeated_frame_tx.shape[0], 1) + 1j * np.random.rand(repeated_frame_tx.shape[0],
                                                                                            1)
                a_phase = a_amp / np.abs(a_amp)  # pure phase shift

                cfo_arr = a_phase * np.exp(1j * np.arange(0, repeated_frame_tx.shape[1], 1) * cfo_set)
                data_rx_dummy = data_rx_dummy * cfo_arr

                data = data_rx_dummy
            else:
                # actual SDR transmission
                data = sdr.rx()

            ber_c, snr_c, rho_avg_plot, evm_arr = receiver_MIMO(data, mimo, inPar.Ntx, pilot_rep_use)

            new_SNR_guard = snr_c
            new_BER = np.mean(np.array(ber_c).flatten())
            new_EVM = np.mean(np.array(evm_arr).flatten())
                # end dummpy Rx

            # end update Rx status


            #new_temp = 25 + 5 * np.sin(self.counter * 0.1) + random.uniform(-0.5, 0.5)
            #new_humidity = 50 + 20 * np.sin(self.counter * 0.05) + random.uniform(-1, 1)
            #new_pressure = 1000 + 20 * np.sin(self.counter * 0.03) + random.uniform(-0.5, 0.5)

            # Update data structures
            self.time_points.append(self.counter)
            self.data['SNRguard'].append(new_SNR_guard)
            self.data['BER'].append(new_BER)
            self.data['EVM'].append(new_EVM)

            self.counter += 1
            time.sleep(0.3)  # Simulate 0.1 second delay

    def on_closing(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = RxGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()