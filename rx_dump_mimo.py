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

cfg_test =[(102, 'MMSE_rep4'), (2, 'MMSE_rep4')]
Ntx = 1

# path to dataset
#DUMP_FILE = Path("dump_last_10_frames_2GHZ_40m_chair.pkl")
#DUMP_FILE_list = [Path("dump_last_10_frames_p1.pkl"), Path("dump_last_10_frames_p2.pkl")]
DUMP_FILE_list = [Path("dump_last_10_frames_pp1.pkl"), Path("dump_last_10_frames_pp2.pkl")]

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

        ber_c, snr_c, rho_avg_plot, evm_arr, Rhh = receiver_MIMO_v2(data, mimo, Ntx, pilot_rep_use)

        Rhh_list.append(Rhh)

        if idx == 0:
            evm_bl = evm_arr
            print(f'Idx{preloaded_frame_idx}: ber={ber_c} evm={evm_arr} MimoMode={mimo}')
        else:
            gain_evm = evm_arr - evm_bl
            gain_avg.append(np.mean(gain_evm))
            print(f'Idx{preloaded_frame_idx}: ber={ber_c} evm={evm_arr} MimoMode={mimo} EVMgain={gain_evm}')

print(f'Average EVM gain = {np.mean(gain_avg)}')

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




