"""
SDR MIMO OFDM analysis: Nrx=2 receive antennas, preamble + 4 pilots + Ndata data symbols.
Tests channel estimation (CE) algorithms on captured baseband; target metric: ber_c (bit error rate).
OFDM demodulation, sync, and CFO correction are unchanged; only CE and equalization options are varied.

Channel estimation improvements (in rx_funcs_mimo):
  - ce_mode: 0=time-window IFFT/FFT, 1=DFT-based, 2=DFT+MMSE reg, 3=DFT+SVD+MMSE.
  - CE_mode=4: ESPRIT super-resolution (exact delay taps) + MMSE projection; n_taps_esprit = number of taps.
  - CE_mode=5: joint Rx-antenna DFT-based CE.
  - CE_mode=6: joint Rx-antenna DNN (Transformer) CE — loads transformer_ce_best_2rx1tx.pt.
  - Adaptive DC: dc_adaptive_threshold > 0 + dc_mask_half_width > 0 → interpolate DC region only when
    DC power exceeds threshold × median(other subcarriers); avoids BER loss when no spike present.
  - exclude_dc_from_ruu: exclude DC bin from pilot residual when computing Ruu (stops spike inflating noise cov).
  - robust_pilot_avg: median over pilot repeats instead of mean (reduces outlier pilot symbols).
"""
import re
import pickle
from pathlib import Path

import adi
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import scipy
import time
import pylab as pl
from IPython import display

import sionna as sn
import tensorflow as tf

#from tx_mimo import create_data

from matplotlib.animation import FuncAnimation

from tx_mimo_npilot import create_data, create_preamble, create_data_frame, init_tx_dict
from rx_funcs_mimo import receiver_MIMO_v2

from system_tx import SysParUL
import json

# (mimo_mode, label): mimo_mode digits = [CE_mode*100] + [SMMSE_mode*10] + MIMO_det (e.g. 102 = CE=1, MIMO=2)
cfg_test = [(502, 'MMSE_rep4'), (602, 'DNN_joint_rep4')]
# DC handling: adaptive avoids BER loss when no spike; excluding DC from Ruu often helps most
CE_DC_MASK_HALF_WIDTH = 1       # width for DC interpolation when adaptive triggers (0=no CE DC correction)
CE_DC_ADAPTIVE_THRESHOLD = 3.0   # 0=off. If DC power > this × median(other), interpolate DC region
EXCLUDE_DC_FROM_RUU = True       # exclude DC bin from pilot residual when computing Ruu (recommended)
CE_ROBUST_PILOT_AVG = False      # True = median over pilot repeats (robust to outlier pilots)
N_TAPS_ESPRIT = 1            # CE_mode=4: number of delay taps estimated by ESPRIT (super-resolution)
PLOT_CHANNEL_FREQ = False         # plot frequency channel before CE (LS, comb) vs after CE (with legend CE_mode)
TURBO_ENABLE = False             # Turbo receiver: decision-directed channel refinement
TURBO_ITERS = 1                  # Number of turbo refinement iterations
TURBO_PILOT_WEIGHT = 0.5         # Weight for pilot LS vs data-aided LS in turbo CE (0..1)
Ntx = 1


def plot_channel_freq(h_ls_before_full, h_ce, ce_mode, frame_idx=0, rec_name=''):
    """Plot frequency channel: before CE (LS on pilot subcarriers only) vs after CE (full)."""
    # h_ls_before_full, h_ce: (iNtx, num_rx, N_sc_use)
    n_tx, n_rx, n_sc = h_ls_before_full.shape
    sc_idx = np.arange(n_sc)
    ce_label = f"CE_mode={ce_mode}"
    for tx in range(n_tx):
        for rx in range(n_rx):
            fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
            # After CE: full response
            h_after = np.asarray(h_ce[tx, rx, :], dtype=np.complex128)
            ax_mag.plot(sc_idx, np.abs(h_after), '-', label=f'After CE ({ce_label})', color='C0')
            ax_phase.plot(sc_idx, np.angle(h_after), '-', label=f'After CE ({ce_label})', color='C0')
            # Before CE: only at pilot positions (comb); NaN elsewhere
            h_before = np.asarray(h_ls_before_full[tx, rx, :], dtype=np.complex128)
            valid = ~np.isnan(np.real(h_before))
            if np.any(valid):
                ax_mag.scatter(sc_idx[valid], np.abs(h_before[valid]), s=18, label='Before CE (LS, comb)', color='C1', zorder=3)
                ax_phase.scatter(sc_idx[valid], np.angle(h_before[valid]), s=18, label='Before CE (LS, comb)', color='C1', zorder=3)
            ax_mag.set_ylabel('|H|')
            ax_mag.legend(loc='upper right', fontsize=8)
            ax_mag.grid(True, alpha=0.3)
            ax_phase.set_ylabel('angle(H) [rad]')
            ax_phase.set_xlabel('Subcarrier index')
            ax_phase.legend(loc='upper right', fontsize=8)
            ax_phase.grid(True, alpha=0.3)
            fig.suptitle(f'Channel freq: tx={tx} rx={rx}  frame={frame_idx}  {rec_name}')
            plt.tight_layout()
            plt.show()

# path to dataset
#DUMP_FILE = Path("dump_last_10_frames_2GHZ_40m_chair.pkl")
#DUMP_FILE_list = [Path("dump_last_10_frames_p1.pkl"), Path("dump_last_10_frames_p2.pkl")]
#DUMP_FILE_list = [Path("dump_last_10_frames_pp1.pkl"), Path("dump_last_10_frames_pp2.pkl")] # set QAM256 in sys cfg

DUMP_FILE_list = [Path("dump_last_10_frames_20260421_124438.pkl")]

preloaded_frames = list()
for DUMP_FILE in DUMP_FILE_list:
    if DUMP_FILE.exists():
        with DUMP_FILE.open("rb") as dump_fd:
            preloaded_frames_c = pickle.load(dump_fd)
        if len(preloaded_frames_c) == 0:
            raise ValueError(f"{DUMP_FILE} is empty. Capture frames with YES first.")
    else:
        raise Exception('Dump file does not exist.')


    preloaded_frames_c = [np.array(frame) for frame in preloaded_frames_c]
    preloaded_frames = preloaded_frames + preloaded_frames_c



use_preloaded_frames = True
preloaded_frame_idx = 0
data = preloaded_frames[0]

gain_avg = list()
Rhh_list = list()

for preloaded_frame_idx in range(len(preloaded_frames)):
    ### SDR reception
    data = preloaded_frames[preloaded_frame_idx]

    evm_bl = np.zeros(Ntx)

    for idx, (mimo, rec_name) in enumerate(cfg_test):

        # parse configs
        try:
            pilot_rep_use = int(re.findall('_rep(\d+)', rec_name)[0])
        except:
            pilot_rep_use = 1

        if PLOT_CHANNEL_FREQ:
            ber_c, snr_c, rho_avg_plot, evm_arr, Rhh, h_ls_before_full, h_ce_full, ce_mode_used = receiver_MIMO_v2(
                data, mimo, Ntx, pilot_rep_use,
                dc_mask_half_width=CE_DC_MASK_HALF_WIDTH,
                dc_adaptive_threshold=CE_DC_ADAPTIVE_THRESHOLD,
                robust_pilot_avg=CE_ROBUST_PILOT_AVG,
                exclude_dc_from_ruu=EXCLUDE_DC_FROM_RUU,
                n_taps_esprit=N_TAPS_ESPRIT,
                return_channel_for_plot=True,
                turbo_enable=TURBO_ENABLE,
                turbo_iters=TURBO_ITERS,
                turbo_pilot_weight=TURBO_PILOT_WEIGHT,
            )
            if preloaded_frame_idx == 0:
                plot_channel_freq(h_ls_before_full, h_ce_full, ce_mode_used, frame_idx=0, rec_name=rec_name)
        else:
            ber_c, snr_c, rho_avg_plot, evm_arr, Rhh = receiver_MIMO_v2(
                data, mimo, Ntx, pilot_rep_use,
                dc_mask_half_width=CE_DC_MASK_HALF_WIDTH,
                dc_adaptive_threshold=CE_DC_ADAPTIVE_THRESHOLD,
                robust_pilot_avg=CE_ROBUST_PILOT_AVG,
                exclude_dc_from_ruu=EXCLUDE_DC_FROM_RUU,
                n_taps_esprit=N_TAPS_ESPRIT,
                turbo_enable=TURBO_ENABLE,
                turbo_iters=TURBO_ITERS,
                turbo_pilot_weight=TURBO_PILOT_WEIGHT,
            )

        Rhh_list.append(Rhh)

        if idx == 0:
            evm_bl = evm_arr
            print(f'Idx{preloaded_frame_idx}: ber={ber_c} evm={evm_arr} MimoMode={mimo}')
        else:
            gain_evm = evm_arr - evm_bl
            gain_avg.append(np.mean(gain_evm))
            print(f'Idx{preloaded_frame_idx}: ber={ber_c} evm={evm_arr} MimoMode={mimo} EVMgain={gain_evm}')

print(f'Average EVM gain = {np.mean(gain_avg)}')

if False:
    Rhh0 = Rhh_list[0]
    rhh_flat = Rhh0.flatten()
    for idx in range(len(Rhh_list)):
        Rhh_c = Rhh_list[idx]
        condRhh = np.linalg.cond(Rhh_c)
        rhh_flat_c = Rhh_c.flatten()

        #corr = np.abs( np.vdot(rhh_flat, rhh_flat_c) / np.linalg.norm(rhh_flat)  /  np.linalg.norm(rhh_flat_c))
        corr = np.vdot(rhh_flat, rhh_flat_c) / np.linalg.norm(rhh_flat) / np.linalg.norm(rhh_flat_c)

        D_inv = np.diag(1.0 / np.abs(np.sqrt(np.diag(Rhh_c))))

        Rhh_norm_c = D_inv @ Rhh_c @ D_inv
        print(np.abs(Rhh_norm_c))


        print(f'Idx={idx} corrRhh={corr} condRhh={condRhh}')




