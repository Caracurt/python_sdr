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

cfg_test = (0, 'MMSE_rep4') # set config for reception, it would be MMSE by default set in json

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
        
        # Channel estimation mode
        self.ce_mode = 0  # Default to ce_mode=0
        
        # SMMSE mode
        self.smmse_mode = 0  # Default to smmse_mode=0 (no SMMSE channel estimation)
        
        # MIMO detection mode
        self.mimo_mode = 2  # Default to mimo_mode=2 (MMSE detection mode)

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
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create left frame for plot
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create figure for plotting
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.line, = self.ax.plot([], [], 'b-')
        self.ax.set_title(f"Real-Time {self.current_metric} (Last 20 Points)")
        self.ax.set_xlabel("Most Recent Time Points")
        self.ax.set_ylabel(self.current_metric)
        self.ax.grid(True)

        # Create canvas for matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create right frame for text box and controls
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        # Create text box for averages
        avg_frame = ttk.LabelFrame(right_frame, text="Last 10 Experiments Average")
        avg_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        self.avg_text = tk.Text(avg_frame, height=8, width=30, wrap=tk.WORD)
        self.avg_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Make text box read-only
        self.avg_text.config(state=tk.DISABLED)

        # Create radio buttons for MIMO detection mode
        mimo_frame = ttk.LabelFrame(right_frame, text="MIMO Detection Mode")
        mimo_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        self.mimo_mode_var = tk.IntVar(value=self.mimo_mode)

        # MIMO mode options: 0=SumRx, 1=EigRx, 2=MMSE, 3=IRC, 4=IRC (advanced)
        mimo_mode_labels = {
            0: "Mode 0 (SumRx)",
            1: "Mode 1 (EigRx)",
            2: "Mode 2 (MMSE)",
            3: "Mode 3 (IRC)",
            4: "Mode 4 (IRC Advanced)"
        }

        for mode_value in range(5):
            rb = ttk.Radiobutton(
                mimo_frame,
                text=mimo_mode_labels[mode_value],
                variable=self.mimo_mode_var,
                value=mode_value,
                command=self.mimo_mode_changed
            )
            rb.pack(side=tk.TOP, padx=5, pady=2)

        # Create radio buttons for SMMSE mode
        smmse_frame = ttk.LabelFrame(right_frame, text="SMMSE Mode")
        smmse_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        self.smmse_mode_var = tk.IntVar(value=self.smmse_mode)

        smmse_mode_0 = ttk.Radiobutton(
            smmse_frame,
            text="No SMMSE (0)",
            variable=self.smmse_mode_var,
            value=0,
            command=self.smmse_mode_changed
        )
        smmse_mode_0.pack(side=tk.TOP, padx=5, pady=2)

        smmse_mode_1 = ttk.Radiobutton(
            smmse_frame,
            text="Spatial Channel Estimation (1)",
            variable=self.smmse_mode_var,
            value=1,
            command=self.smmse_mode_changed
        )
        smmse_mode_1.pack(side=tk.TOP, padx=5, pady=2)

        # Create radio buttons for channel estimation mode
        ce_frame = ttk.LabelFrame(right_frame, text="Channel Estimation Mode")
        ce_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        self.ce_mode_var = tk.IntVar(value=self.ce_mode)

        ce_mode_0 = ttk.Radiobutton(
            ce_frame,
            text="CE Mode 0",
            variable=self.ce_mode_var,
            value=0,
            command=self.ce_mode_changed
        )
        ce_mode_0.pack(side=tk.TOP, padx=5, pady=2)

        ce_mode_1 = ttk.Radiobutton(
            ce_frame,
            text="CE Mode 1",
            variable=self.ce_mode_var,
            value=1,
            command=self.ce_mode_changed
        )
        ce_mode_1.pack(side=tk.TOP, padx=5, pady=2)

        # Create radio buttons for metric selection
        self.metric_var = tk.StringVar(value=self.current_metric)

        metrics_frame = ttk.LabelFrame(right_frame, text="Select Metric")
        metrics_frame.pack(side=tk.BOTTOM, fill=tk.X)

        for metric in self.data.keys():
            rb = ttk.Radiobutton(
                metrics_frame,
                text=metric,
                variable=self.metric_var,
                value=metric,
                command=self.metric_changed
            )
            rb.pack(side=tk.TOP, padx=5, pady=2)

    def calculate_averages(self):
        """Calculate averages for the last 10 experiments for each metric"""
        averages = {}
        for metric in self.data.keys():
            if len(self.data[metric]) >= 10:
                # Get last 10 values
                last_10 = self.data[metric][-10:]
                averages[metric] = np.mean(last_10)
            elif len(self.data[metric]) > 0:
                # If less than 10 values, use all available
                averages[metric] = np.mean(self.data[metric])
            else:
                averages[metric] = 0.0
        return averages

    def update_averages_display(self):
        """Update the text box with current averages"""
        averages = self.calculate_averages()
        
        # Enable text widget for editing
        self.avg_text.config(state=tk.NORMAL)
        
        # Clear existing content
        self.avg_text.delete(1.0, tk.END)
        
        # Add header
        self.avg_text.insert(tk.END, "Last 10 Experiments Average:\n")
        self.avg_text.insert(tk.END, "=" * 30 + "\n\n")
        
        # Add averages for each metric
        for metric, avg_value in averages.items():
            if metric == 'SNRguard':
                self.avg_text.insert(tk.END, f"SNR Guard: {avg_value:.2f} dB\n")
            elif metric == 'BER':
                self.avg_text.insert(tk.END, f"BER: {avg_value:.4f}\n")
            elif metric == 'EVM':
                self.avg_text.insert(tk.END, f"EVM: {avg_value:.4f}\n")
        
        # Disable text widget to make it read-only
        self.avg_text.config(state=tk.DISABLED)

    def mimo_mode_changed(self):
        """Handle MIMO detection mode change"""
        self.mimo_mode = self.mimo_mode_var.get()
        # The change will take effect on the next sensor reading

    def smmse_mode_changed(self):
        """Handle SMMSE mode change"""
        self.smmse_mode = self.smmse_mode_var.get()
        # The change will take effect on the next sensor reading

    def ce_mode_changed(self):
        """Handle channel estimation mode change"""
        self.ce_mode = self.ce_mode_var.get()
        # The change will take effect on the next sensor reading

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
            self.update_averages_display()  # Update averages display
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

            ber_c, snr_c, rho_avg_plot, evm_arr = receiver_MIMO(data, self.mimo_mode, inPar.Ntx, pilot_rep_use, ce_mode=self.ce_mode, smmse_mode=self.smmse_mode)

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