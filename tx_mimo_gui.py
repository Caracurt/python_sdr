"""
GUI MIMO OFDM transmit (same baseband as tx_mimo_npilot.py).

Modes (user-selectable):
- Cyclic TX (tx_cyclic_buffer=True): one sdr.tx(); hardware repeats the buffer
  gaplessly until Stop.
- Non-cyclic TX (tx_cyclic_buffer=False): worker loops sdr.tx(buffer) until Stop.
  Small IQ gaps can occur between pushes; TX gain is still updated whenever the
  slider moves (Scale command callback -> _apply_tx_gain), sharing _sdr_lock with
  the transmit loop.

Stop always uses tx_destroy_buffer() when available.
"""
from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

import adi
from tx_mimo_npilot import create_data_frame, init_tx_dict

# Override URIs here to match your network (rx_gui uses ad9361 @ 192.168.2.2).
PLUTO_URI = "ip:192.168.1.1"
ANTS_DR_URI = "ip:192.168.1.1"


@dataclass
class TxBufferInfo:
    """Prepared complex baseband for sdr.tx (same scaling as tx_mimo_npilot.main)."""

    tx_buffer: np.ndarray
    frame_size: int
    is_pluto: bool
    rx_buffer_num_samps: int  # len(repeated_frame), for ANTsdr rx_buffer_size (matches npilot)


def build_tx_buffer(inPar) -> TxBufferInfo:
    (repeated_frame_tx, repeated_frame, _frame_len, _preamble_len, _preamble_core,
     _mod_dict_data, _pilot_tx, _bite_stream_tx, _mod_data, _bite_stream_tx_uncode) = (
        create_data_frame(inPar)
    )
    rx_buffer_num_samps = int(len(repeated_frame))
    frame_size = repeated_frame_tx.shape[0]
    if inPar.Ntx > 1:
        repeated_frame_tx = repeated_frame_tx.T
    else:
        repeated_frame_tx = repeated_frame[:, 0]

    if inPar.device_type != "ANTsdr":
        tx_buffer = (repeated_frame[:, 0] * (2**16)).astype(np.complex64)
        return TxBufferInfo(
            tx_buffer=tx_buffer,
            frame_size=int(frame_size),
            is_pluto=True,
            rx_buffer_num_samps=rx_buffer_num_samps,
        )

    tmp_arr = np.asarray(repeated_frame_tx)
    tmp_arr1 = tmp_arr.flatten()
    max_el = float(np.max(np.abs(np.hstack((np.real(tmp_arr1), np.imag(tmp_arr1))))))
    scale = (1.0 / max_el) * (2**15)
    tx_buffer = (repeated_frame_tx * scale).astype(np.complex64)
    return TxBufferInfo(
        tx_buffer=tx_buffer,
        frame_size=int(frame_size),
        is_pluto=False,
        rx_buffer_num_samps=rx_buffer_num_samps,
    )


def create_sdr(inPar, buf: TxBufferInfo, *, cyclic: bool) -> Any:
    """Configure SDR like tx_mimo_npilot.main(); cyclic selects tx_cyclic_buffer."""
    if buf.is_pluto:
        sdr = adi.Pluto(PLUTO_URI)
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0 = 70.0
        sdr.rx_lo = int(inPar.center_freq)
        sdr.sample_rate = int(inPar.sample_rate)
        sdr.rx_rf_bandwidth = int(inPar.sample_rate)
        sdr.rx_buffer_size = buf.frame_size
        sdr.tx_rf_bandwidth = int(inPar.sample_rate)
        sdr.tx_lo = int(inPar.center_freq)
        sdr.tx_cyclic_buffer = cyclic
        sdr.tx_buffer_size = buf.frame_size
        return sdr

    sdr = adi.ad9363(uri=ANTS_DR_URI)
    num_samps = buf.rx_buffer_num_samps
    rx_lo = int(inPar.center_freq)
    rx_mode = "slow_attack"
    rx_gain0 = 70
    rx_gain1 = 70
    sdr.tx_enabled_channels = list(range(inPar.Ntx))
    sdr.sample_rate = int(inPar.sample_rate)
    sdr.rx_lo = int(rx_lo)
    sdr.gain_control_mode = rx_mode
    sdr.rx_hardwaregain_chan0 = int(rx_gain0)
    sdr.rx_hardwaregain_chan1 = int(rx_gain1)
    sdr.rx_buffer_size = int(num_samps)
    sdr.tx_rf_bandwidth = int(inPar.sample_rate)
    sdr.tx_lo = int(inPar.center_freq)
    sdr.tx_cyclic_buffer = cyclic
    sdr.tx_buffer_size = buf.frame_size
    return sdr


class TxMimoGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("MIMO OFDM TX (GUI)")

        self.inPar = init_tx_dict()
        self.buf_info = build_tx_buffer(self.inPar)

        self.sdr: Optional[Any] = None
        self._sdr_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._tx_thread: Optional[threading.Thread] = None
        self._last_tx_error: Optional[BaseException] = None
        self._tx_use_cyclic: bool = True
        self._tx_lo_hz: int = 2_000_000_000

        self._setup_ui()

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        row0 = ttk.Frame(main)
        row0.pack(fill=tk.X, pady=(0, 8))
        self.btn_start = ttk.Button(row0, text="Start transmission", command=self._on_start)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_stop = ttk.Button(row0, text="Stop", command=self._on_stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 16))

        self.ind_label = tk.Label(row0, text="TX OFF", width=10, fg="white", bg="#666666")
        self.ind_label.pack(side=tk.LEFT)

        lo_fr = ttk.LabelFrame(main, text="LO frequency")
        lo_fr.pack(fill=tk.X, pady=(0, 8))
        self.tx_lo_var = tk.IntVar(value=2_000_000_000)
        rb_2g = ttk.Radiobutton(
            lo_fr,
            text="2.0 GHz (default)",
            variable=self.tx_lo_var,
            value=2_000_000_000,
            command=self._on_tx_lo_changed,
        )
        rb_409m = ttk.Radiobutton(
            lo_fr,
            text="409 MHz",
            variable=self.tx_lo_var,
            value=409_000_000,
            command=self._on_tx_lo_changed,
        )
        rb_2g.pack(side=tk.TOP, anchor=tk.W, padx=8, pady=2)
        rb_409m.pack(side=tk.TOP, anchor=tk.W, padx=8, pady=2)

        mode_fr = ttk.LabelFrame(main, text="TX buffer mode")
        mode_fr.pack(fill=tk.X, pady=(0, 8))
        self.cyclic_tx_var = tk.BooleanVar(value=True)
        self.chk_cyclic = ttk.Checkbutton(
            mode_fr,
            text="Cyclic TX (gapless IQ; hardware repeats one buffer until Stop)",
            variable=self.cyclic_tx_var,
        )
        self.chk_cyclic.pack(anchor=tk.W, padx=8, pady=4)
        ttk.Label(
            mode_fr,
            text="Off: non-cyclic — software loops sdr.tx(); gain still updates on slider move "
            "(may see brief SNR dips between buffers).",
            wraplength=480,
            font=("TkDefaultFont", 8),
        ).pack(anchor=tk.W, padx=8, pady=(0, 6))

        gain_fr = ttk.LabelFrame(main, text="TX gain (dB)")
        gain_fr.pack(fill=tk.X, pady=(0, 8))
        # IntVar + reading Scale/command args avoids stale DoubleVar.get() vs knob during drag.
        self.gain_var = tk.IntVar(value=-10)
        self.gain_scale = tk.Scale(
            gain_fr,
            from_=-70,
            to=0,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.gain_var,
            length=320,
            showvalue=True,
            command=self._on_gain_moved,
        )
        self.gain_scale.pack(fill=tk.X, padx=8, pady=4)
        self.gain_value_lbl = ttk.Label(gain_fr, text="Gain: -10 dB")
        self.gain_value_lbl.pack(anchor=tk.W, padx=8, pady=(0, 4))

        path = "Pluto" if self.buf_info.is_pluto else "ANTsdr (ad9363)"
        dummy = bool(self.inPar.dummyTx)
        self.status_lbl = ttk.Label(
            main,
            text=f"Device path: {path} | dummyTx={dummy} | frame_size={self.buf_info.frame_size}",
            wraplength=400,
        )
        self.status_lbl.pack(fill=tk.X, pady=(4, 0))

    def _set_indicator(self, on: bool) -> None:
        if on:
            self.ind_label.config(text="TX ON", bg="#228822")
        else:
            self.ind_label.config(text="TX OFF", bg="#666666")

    def _set_tx_lo(self, hz: int) -> None:
        hz_i = int(hz)
        self._tx_lo_hz = hz_i
        if self.sdr is None:
            return
        with self._sdr_lock:
            self.sdr.tx_lo = hz_i

    def _restart_cyclic_tx_if_running(self) -> None:
        if self.sdr is None:
            return
        if self._tx_thread is None or (not self._tx_thread.is_alive()):
            return
        if not self._tx_use_cyclic:
            return
        buf = self.buf_info.tx_buffer
        with self._sdr_lock:
            if hasattr(self.sdr, "tx_destroy_buffer"):
                self.sdr.tx_destroy_buffer()
            self.sdr.tx_lo = int(self._tx_lo_hz)
            self.sdr.tx(buf)

    def _on_tx_lo_changed(self) -> None:
        hz = int(self.tx_lo_var.get())
        self._set_tx_lo(hz)
        # If we're currently transmitting in cyclic mode, briefly restart the DMA
        # buffer so the new LO takes effect cleanly.
        self._restart_cyclic_tx_if_running()

    def _read_gain_db_int(self) -> int:
        """Current slider value in dB, clamped [-70, 0]. Prefer widget .get() (authoritative)."""
        try:
            v = float(self.gain_scale.get())
        except (tk.TclError, TypeError, ValueError):
            try:
                v = float(self.gain_var.get())
            except (tk.TclError, TypeError, ValueError):
                v = -10.0
        g = int(round(v))
        return max(-70, min(0, g))

    def _apply_tx_gain(self, db: int | float) -> None:
        g = int(round(float(db)))
        g = max(-70, min(0, g))
        if self.sdr is None:
            return
        with self._sdr_lock:
            if self.buf_info.is_pluto:
                self.sdr.tx_hardwaregain_chan0 = g
            else:
                self.sdr.tx_hardwaregain_chan0 = g
                if self.inPar.Ntx > 1:
                    self.sdr.tx_hardwaregain_chan1 = g

    def _on_gain_moved(self, *args: object) -> None:
        # Tk passes the new scale value as the first argument to command=...; use it so we
        # never apply a stale value while the knob has already moved (variable can lag).
        db: float
        if args:
            try:
                db = float(str(args[0]))
            except (TypeError, ValueError):
                db = float(self.gain_scale.get())
        else:
            db = float(self.gain_scale.get())
        g = max(-70, min(0, int(round(db))))
        self.gain_value_lbl.config(text=f"Gain: {g} dB")
        if self.sdr is not None:
            self._apply_tx_gain(g)

    def _on_start(self) -> None:
        if self._tx_thread is not None and self._tx_thread.is_alive():
            return
        self._stop_event.clear()

        if self.inPar.dummyTx:
            self.sdr = None
            self._set_indicator(True)
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.chk_cyclic.config(state=tk.DISABLED)
            self._tx_thread = threading.Thread(target=self._tx_worker_dummy, daemon=True)
            self._tx_thread.start()
            return

        self._tx_use_cyclic = bool(self.cyclic_tx_var.get())
        try:
            self.sdr = create_sdr(self.inPar, self.buf_info, cyclic=self._tx_use_cyclic)
            self._set_tx_lo(int(self.tx_lo_var.get()))
            self._apply_tx_gain(self._read_gain_db_int())
        except Exception as e:
            self.status_lbl.config(text=f"SDR open/config failed: {e}")
            self.sdr = None
            return

        self._set_indicator(True)
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.chk_cyclic.config(state=tk.DISABLED)
        if self._tx_use_cyclic:
            self.status_lbl.config(text="Transmitting (cyclic DMA — continuous until Stop)…")
            target = self._tx_worker_cyclic
        else:
            self.status_lbl.config(text="Transmitting (non-cyclic loop; change gain with slider)…")
            target = self._tx_worker_noncyclic
        self._tx_thread = threading.Thread(target=target, daemon=True)
        self._tx_thread.start()

    def _tx_worker_dummy(self) -> None:
        while not self._stop_event.is_set():
            #time.sleep(0.05)
            time.sleep(0.0001)

    def _tx_worker_cyclic(self) -> None:
        assert self.sdr is not None
        buf = self.buf_info.tx_buffer
        self._last_tx_error = None
        try:
            with self._sdr_lock:
                self.sdr.tx(buf)
            self._stop_event.wait()
        except BaseException as e:
            self._last_tx_error = e
        finally:
            try:
                self.root.after(0, self._tx_thread_finished)
            except tk.TclError:
                pass

    def _tx_worker_noncyclic(self) -> None:
        """Repeat buffer until Stop; slider command updates gain (same _sdr_lock)."""
        assert self.sdr is not None
        buf = self.buf_info.tx_buffer
        self._last_tx_error = None
        try:
            while not self._stop_event.is_set():
                with self._sdr_lock:
                    self.sdr.tx(buf)
        except BaseException as e:
            self._last_tx_error = e
        finally:
            try:
                self.root.after(0, self._tx_thread_finished)
            except tk.TclError:
                pass

    def _tx_thread_finished(self) -> None:
        """Main-thread follow-up when the TX worker exits (error path or after join + user cleanup)."""
        if self.inPar.dummyTx:
            return
        if self.sdr is not None:
            self._tx_finished_cleanup(err=self._last_tx_error)

    def _tx_finished_cleanup(self, err: Optional[BaseException] = None) -> None:
        self._set_indicator(False)
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        try:
            self.chk_cyclic.config(state=tk.NORMAL)
        except tk.TclError:
            pass
        if self.sdr is not None and not self.inPar.dummyTx:
            try:
                if hasattr(self.sdr, "tx_destroy_buffer"):
                    self.sdr.tx_destroy_buffer()
            except Exception:
                pass
            self.sdr = None
        if err is not None:
            self.status_lbl.config(text=f"TX error: {err}")
        else:
            self.status_lbl.config(
                text=f"Stopped. Device path: {'Pluto' if self.buf_info.is_pluto else 'ANTsdr'} | "
                f"frame_size={self.buf_info.frame_size}"
            )

    def _on_stop(self) -> None:
        self._stop_event.set()
        if self._tx_thread is not None:
            self._tx_thread.join(timeout=5.0)
            self._tx_thread = None
        if self.inPar.dummyTx:
            self._set_indicator(False)
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            try:
                self.chk_cyclic.config(state=tk.NORMAL)
            except tk.TclError:
                pass
            return
        if self.sdr is not None:
            err = self._last_tx_error
            self._tx_finished_cleanup(err=err)

    def _on_close(self) -> None:
        self._on_stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    TxMimoGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
