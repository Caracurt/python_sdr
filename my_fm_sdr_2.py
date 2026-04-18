# FM radio receiver via SDR (AD9361): real-time receive -> FM demod -> LPF+decimate -> audio
# Target: 36.5 MHz. Play for DURATION_SEC using producer-consumer audio.
import argparse
import queue
import threading
import time

import adi
import numpy as np
import scipy.signal
import sounddevice as sd

import scipy
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal

# get audio
filename = '01-We-Die-Young.wav'
data, fs = sf.read(filename)

fc = 2e9
up_s = 10

# --- Parameters (overridable via argparse) ---
DURATION_SEC = 60
oversampling = up_s
cut_off_hz = 15_000


RX_BUFFER_SIZE = 4096
SDR_URI = "ip:192.168.2.2"

# Audio (48 kHz sink)
SAMPLE_RATE = fs
BLOCKSIZE = 1024
QUEUE_MAXSIZE = 30


def fm_demod_phase_diff(samples):
    """FM demod: phase difference between consecutive samples. Returns real array length len(samples)-1."""
    return np.angle(samples[1:] * np.conj(samples[:-1]))


def build_lpf(cut_off_hz, fs, order=5):
    """Butterworth LPF (b, a) for use with lfilter."""
    return scipy.signal.butter(order, cut_off_hz, fs=fs, btype="low")


def run_fm_receiver(duration_sec, oversamp, cut_off, sdr_uri):
    sdr_sample_rate = int(SAMPLE_RATE * oversamp)
    b, a = build_lpf(cut_off, sdr_sample_rate)

    sdr = adi.ad9361(uri=sdr_uri)
    sdr.sample_rate = sdr_sample_rate
    sdr.rx_lo = int(fc)
    sdr.rx_buffer_size = RX_BUFFER_SIZE
    sdr.rx_enabled_channels = [0]
    sdr.gain_control_mode = "slow_attack"
    sdr.rx_hardwaregain_chan0 = 70

    audio_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    SENTINEL = object()
    remainder = [np.array([]).reshape(0, 1)]
    remainder_lock = threading.Lock()
    stop_requested = [False]
    total_queued = [0]

    def callback(outdata, frames, time_info, status):
        if status:
            print(status)
        with remainder_lock:
            need = frames
            out = np.zeros((frames, 1), dtype=np.float32)
            written = 0
            while written < frames:
                rem = remainder[0]
                if rem.shape[0] >= need:
                    out[written : written + need] = rem[:need]
                    remainder[0] = rem[need:]
                    written += need
                    break
                if rem.shape[0] > 0:
                    n = rem.shape[0]
                    out[written : written + n] = rem
                    written += n
                    need -= n
                    remainder[0] = np.array([]).reshape(0, 1)
                try:
                    chunk = audio_queue.get_nowait()
                except queue.Empty:
                    break
                if chunk is SENTINEL:
                    stop_requested[0] = True
                    break
                rem = np.asarray(chunk, dtype=np.float32)
                if rem.ndim == 1:
                    rem = rem.reshape(-1, 1)
                remainder[0] = rem
            if written < frames:
                out[written:] = 0
            outdata[:] = out
        if stop_requested[0]:
            raise sd.CallbackStop

    def producer():
        start = time.monotonic()
        while (time.monotonic() - start) < duration_sec:
            # raw = np.array(sdr.rx()).flatten()
            # demod = fm_demod_phase_diff(raw)
            # filtered = scipy.signal.lfilter(b, a, demod)
            # decimated = filtered[::oversamp].astype(np.float32)
            # chunk = decimated.reshape(-1, 1)

            # new demod
            data = sdr.rx()
            data = np.array(data)

            phase_rx = np.angle(data[1:] * np.conj(data[0:-1]))

            data_cut_down1 = signal.resample_poly(phase_rx, 1, up_s)

            chunk = data_cut_down1.reshape(-1, 1)

            try:
                audio_queue.put(chunk, block=True, timeout=5.0)
                total_queued[0] += chunk.shape[0]
            except queue.Full:
                break
        audio_queue.put(SENTINEL)

    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
        blocksize=BLOCKSIZE,
    ):
        prod = threading.Thread(target=producer)
        prod.start()
        prod.join()
        drain_ms = int(1000 * total_queued[0] / SAMPLE_RATE) + 2000
        sd.sleep(drain_ms)
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FM SDR receiver at 36.5 MHz")
    ap.add_argument("--duration", type=float, default=DURATION_SEC, help="Play duration (seconds)")
    ap.add_argument("--oversampling", type=int, default=oversampling, help="SDR rate = 48k * this")
    ap.add_argument("--cut-off", type=float, default=cut_off_hz, dest="cut_off", help="LPF cutoff (Hz)")
    ap.add_argument("--uri", type=str, default=SDR_URI, help="SDR URI")
    args = ap.parse_args()
    run_fm_receiver(args.duration, args.oversampling, args.cut_off, args.uri)
