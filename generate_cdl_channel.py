"""
3GPP 5G CDL Channel Generator using Sionna 0.13.0

Generates wireless propagation channel according to 3GPP 5G CDL model.
Requires: sionna==0.13.0, tensorflow, numpy
"""
import pickle
import numpy as np
import tensorflow as tf

# Sionna 0.13.0
import sionna as sn
from sionna.channel.tr38901 import CDL, AntennaArray
from sionna.channel.utils import cir_to_ofdm_channel, subcarrier_frequencies

# --- Parameters ---
CARRIER_FREQ = 3.5e9          # Hz
N_OFDM = 12                    # number of OFDM symbols
N_UE = 4                       # number of users
N_UE_ANT = 2                   # UE antennas per user (cross-polarized)
N_BS_ANT = 256                 # BS antenna elements (8 vert x 16 horiz x 2 cross-pol)
N_SC = 192                      # number of subcarriers
SUBCARRIER_SPACING = 30e3      # Hz
CDL_MODEL = 'C'                # CDL-C
DELAY_SPREAD = 100e-9          # 100 ns
UE_SPEED_MIN = 3.0 / 3.6       # 3 km/h in m/s
UE_SPEED_MAX = 5.0 / 3.6       # 5 km/h in m/s
OUTPUT_PKL = "cdl_channel_Hfr.pkl"

# BS: 8 vertical x 16 horizontal x 2 cross-pol = 256 elements
# (4x16x2=128; 8x16x2=256 to meet 256 total)
# vertical_spacing=1 lambda, horizontal_spacing=0.5 lambda, pattern 38.901
BS_NUM_ROWS = 8
BS_NUM_COLS = 16
BS_VERT_SPACING = 1.0
BS_HORIZ_SPACING = 0.5

# UE: 1 x 2 cross-pol = 2 elements
UE_NUM_ROWS = 1
UE_NUM_COLS = 2
UE_VERT_SPACING = 0.5
UE_HORIZ_SPACING = 0.5


def create_bs_array():
    return AntennaArray(
        num_rows=BS_NUM_ROWS,
        num_cols=BS_NUM_COLS,
        polarization='dual',
        polarization_type='VH',
        antenna_pattern='38.901',
        carrier_frequency=CARRIER_FREQ,
        vertical_spacing=BS_VERT_SPACING,
        horizontal_spacing=BS_HORIZ_SPACING,
    )


def create_ue_array():
    return AntennaArray(
        num_rows=UE_NUM_ROWS,
        num_cols=UE_NUM_COLS,
        polarization='dual',
        polarization_type='VH',
        antenna_pattern='38.901',
        carrier_frequency=CARRIER_FREQ,
        vertical_spacing=UE_VERT_SPACING,
        horizontal_spacing=UE_HORIZ_SPACING,
    )


def main():
    print("Sionna version:", sn.__version__)
    sampling_frequency = N_SC * SUBCARRIER_SPACING

    bs_array = create_bs_array()
    ue_array = create_ue_array()

    # CDL channel model
    cdl = CDL(
        model=CDL_MODEL,
        delay_spread=DELAY_SPREAD,
        carrier_frequency=CARRIER_FREQ,
        ut_array=ue_array,
        bs_array=bs_array,
        direction='uplink',
        min_speed=UE_SPEED_MIN,
        max_speed=UE_SPEED_MAX,
    )

    # Subcarrier frequencies for cir_to_ofdm_channel
    frequencies = subcarrier_frequencies(N_SC, SUBCARRIER_SPACING)

    # Generate channel for each user (CDL is point-to-point)
    Hfr_list = []
    for u in range(N_UE):
        a, tau = cdl(
            batch_size=1,
            num_time_steps=N_OFDM,
            sampling_frequency=sampling_frequency,
        )
        # a: (1, 1, 256, 1, 4, 24, N_OFDM), tau: (1, 1, 1, 24)
        H_user = cir_to_ofdm_channel(frequencies, a, tau, normalize=False)
        # H_user: (batch, ?, rx_ant, ?, tx_ant, ofdm_symbol, subcarrier)
        # Target: (N_ofdm, N_ue_ant, N_bs_ant, N_sc)
        H_np = tf.squeeze(H_user).numpy()
        Hfr_list.append(H_np)

    # Stack: Hfr_list[u] has shape (rx_ant, tx_ant, ofdm, sc) after squeeze
    # Target per user: (N_ofdm, N_ue_ant, N_bs_ant, N_sc)
    Hfr_list_proc = []
    for u in range(N_UE):
        Hu = np.squeeze(Hfr_list[u])
        # Hu: (256, tx_ant, 10, 72) -> transpose to (10, tx_ant, 256, 72)
        Hu = np.transpose(Hu, (2, 1, 0, 3))
        Hu = Hu[:, :N_UE_ANT, :, :]
        Hfr_list_proc.append(Hu)
    Hfr = np.stack(Hfr_list_proc, axis=1)
    # Hfr: (N_ofdm, N_ue, N_ue_ant, N_bs_ant, N_sc)
    print("Hfr shape:", Hfr.shape)

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(Hfr, f)
    print("Saved to", OUTPUT_PKL)


if __name__ == "__main__":
    main()
