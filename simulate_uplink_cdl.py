"""
Uplink rank-2 wireless link simulator (frequency-domain).
One UE from CDL pkl, 2 antennas, rank-2 transmission to 256-antenna BS.
2 pilot OFDM symbols (even subcarriers, rank-separated) + 10 data OFDM symbols (QPSK).
Channel applied in frequency domain; LS CE with IFFT/window/FFT; MMSE eq; BER (averaged over ranks).
"""
import pickle
import numpy as np
import tensorflow as tf
import sionna as sn
import matplotlib.pyplot as plt

# --- Config ---
PKL_PATH = "cdl_channel_Hfr.pkl"
UE_IDX = 0
N_PILOT_SYM = 2
N_DATA_SYM = 10
RANK = 2
COMB_STEP = 2  # pilot on even subcarriers only
CE_WINDOW_TAPS = 8  # rectangular window: keep first W taps
CE_WIN_LEFT = 4      # keep last WinLeft taps for FFT leakage

# BER vs SNR sweep: SNR from -25 to -12 dB, step 1
SNR_DB_RANGE = np.arange(-25, -10, 2)
CE_MODES = ["ICE", "RCE"]
N_TRIALS = 10  # trials per (SNR, mode) for averaging BER


def load_channel(pkl_path, ue_idx):
    with open(pkl_path, "rb") as f:
        Hfr = pickle.load(f)
    # Hfr: (N_ofdm, N_ue, N_ue_ant, N_bs_ant, N_sc)
    H = Hfr[:, ue_idx, :, :, :]
    return H, Hfr.shape[4], Hfr.shape[0]


def build_pilot_symbols(N_sc, Nsc_pilot):
    inv_sqrt2 = 1.0 / np.sqrt(2)
    rank1_p = np.ones(Nsc_pilot, dtype=np.complex64) * inv_sqrt2
    rank2_p = np.ones(Nsc_pilot, dtype=np.complex64) * inv_sqrt2
    rank1_s2 = np.ones(Nsc_pilot, dtype=np.complex64) * inv_sqrt2
    rank2_s2 = -np.ones(Nsc_pilot, dtype=np.complex64) * inv_sqrt2
    return rank1_p, rank2_p, rank1_s2, rank2_s2


def build_pilot_packet(N_sc, Nsc_pilot, rank1_p, rank2_p, rank1_s2, rank2_s2, scale_pilot):
    x = np.zeros((RANK, N_sc, N_PILOT_SYM + N_DATA_SYM), dtype=np.complex64)
    pilot_sc_idx = np.arange(0, N_sc, COMB_STEP)
    x[0, pilot_sc_idx, 0] = rank1_p * scale_pilot
    x[1, pilot_sc_idx, 0] = rank2_p * scale_pilot
    x[0, pilot_sc_idx, 1] = rank1_s2 * scale_pilot
    x[1, pilot_sc_idx, 1] = rank2_s2 * scale_pilot
    return x, pilot_sc_idx


def build_data_packet(N_sc, scale_data):
    constellation = sn.mapping.Constellation("qam", 2, trainable=False)
    mapper = sn.mapping.Mapper(constellation=constellation)
    num_sym = N_DATA_SYM * N_sc * RANK
    num_bits = num_sym * 2
    bits = np.random.randint(0, 2, (1, num_bits)).astype(np.float32)
    sym = mapper(tf.constant(bits)).numpy().squeeze()
    sym = np.reshape(sym, (N_DATA_SYM, N_sc, RANK), order='F')
    sym = np.transpose(sym, (2, 1, 0))
    return sym * scale_data, bits.squeeze()


def scale_pilot_vs_data(N_sc, Nsc_pilot):
    scale_data = 1.0 / np.sqrt(2)
    total_data = N_DATA_SYM * RANK * N_sc * (scale_data ** 2)
    total_pilot = N_PILOT_SYM * RANK * Nsc_pilot
    scale_pilot = np.sqrt(total_data / total_pilot)
    return scale_pilot, scale_data


def apply_channel(H, x):
    N_sym = x.shape[2]
    N_sc = x.shape[1]
    y = np.zeros((256, N_sc, N_sym), dtype=np.complex64)
    for t in range(N_sym):
        for k in range(N_sc):
            y[:, k, t] = H[t, :, :, k].T @ x[:, k, t]
    return y


def add_awgn(y, snr_db):
    signal_power = np.mean(np.abs(y) ** 2)
    noise_var = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_var / 2) * (np.random.randn(*y.shape) + 1j * np.random.randn(*y.shape))
    return y + noise.astype(np.complex64), noise_var


def ls_ce_pilots(y_bs, pilot_sc_idx, rank1_p, rank2_p, rank1_s2, rank2_s2, scale_pilot):
    Nsc_pilot = len(pilot_sc_idx)
    H_ls = np.zeros((256, RANK, Nsc_pilot), dtype=np.complex64)
    for i_p, k in enumerate(pilot_sc_idx):
        Yp = y_bs[:, k, :N_PILOT_SYM]
        P = np.array([
            [rank1_p[i_p] * scale_pilot, rank1_s2[i_p] * scale_pilot],
            [rank2_p[i_p] * scale_pilot, rank2_s2[i_p] * scale_pilot]
        ], dtype=np.complex64)
        for b in range(256):
            H_ls[b, :, i_p] = np.linalg.solve(P.T, Yp[b, :].T).T
    return H_ls


def ce_interpolate(H_ls, pilot_sc_idx, N_sc, Nsc_pilot):
    N_fft_ce = int(2 ** np.ceil(np.log2(Nsc_pilot)))
    W = min(CE_WINDOW_TAPS, N_fft_ce // 2)
    WinLeft = min(CE_WIN_LEFT, N_fft_ce // 2)
    H_est = np.zeros((256, RANK, N_sc), dtype=np.complex64)
    for b in range(256):
        for r in range(RANK):
            h_f = np.zeros(N_fft_ce, dtype=np.complex64)
            h_f[:Nsc_pilot] = H_ls[b, r, :]
            h_t = np.fft.ifft(h_f, norm='ortho')
            if r == 1:
                peak = np.argmax(np.abs(h_t))
                h_t = np.roll(h_t, -peak)
            # Rectangular window: keep [0:W] and [N_fft_ce-WinLeft:N_fft_ce], zero middle (FFT leakage)
            h_t[W : N_fft_ce - WinLeft] = 0
            h_f2 = np.fft.fft(h_t, norm='ortho')
            h_ref = h_f2[:Nsc_pilot]
            for k in range(N_sc):
                if k <= pilot_sc_idx[0]:
                    H_est[b, r, k] = h_ref[0]
                elif k >= pilot_sc_idx[-1]:
                    H_est[b, r, k] = h_ref[-1]
                else:
                    i = 0
                    while pilot_sc_idx[i + 1] <= k:
                        i += 1
                    w = (k - pilot_sc_idx[i]) / (pilot_sc_idx[i + 1] - pilot_sc_idx[i])
                    H_est[b, r, k] = (1 - w) * h_ref[i] + w * h_ref[i + 1]
    return H_est


def mmse_equalize(y_bs, H_est, noise_var, data_start):
    N_sc = y_bs.shape[1]
    N_data = y_bs.shape[2] - data_start
    s_hat = np.zeros((RANK, N_sc, N_data), dtype=np.complex64)
    for k in range(N_sc):
        H_k = H_est[:, :, k]
        W_k = np.linalg.inv(H_k.conj().T @ H_k + noise_var * np.eye(RANK)) @ H_k.conj().T
        for t in range(N_data):
            y_k = y_bs[:, k, data_start + t]
            s_hat[:, k, t] = W_k @ y_k
    return s_hat


def qpsk_demod_hard(s_hat):
    constellation = sn.mapping.Constellation("qam", 2, trainable=False)
    demapper = sn.mapping.Demapper("maxlog", constellation=constellation)
    bits_list = []
    for r in range(2):
        sr = s_hat[r].T.flatten(order='F')
        sr = sr.reshape(-1, 1)
        llr = demapper([tf.constant(sr), tf.constant(1.0)]).numpy()
        bits_list.append((llr > 0).astype(np.float32).flatten())
    return np.stack(bits_list, axis=0)


def run_trial(H, N_sc, Nsc_pilot, pilot_sc_idx, rank1_p, rank2_p, rank1_s2, rank2_s2,
              scale_pilot, scale_data, snr_db, use_ice):
    """Run one trial: TX, channel, AWGN, CE, equalize, demod, return BER."""
    data_sym, bits_tx = build_data_packet(N_sc, scale_data)
    x, _ = build_pilot_packet(N_sc, Nsc_pilot, rank1_p, rank2_p, rank1_s2, rank2_s2, scale_pilot)
    x[:, :, N_PILOT_SYM:] = data_sym

    y_bs = apply_channel(H, x)
    y_bs, noise_var = add_awgn(y_bs, snr_db)

    H_ls = ls_ce_pilots(y_bs, pilot_sc_idx, rank1_p, rank2_p, rank1_s2, rank2_s2, scale_pilot)
    H_est = ce_interpolate(H_ls, pilot_sc_idx, N_sc, Nsc_pilot)

    if use_ice:
        H_for_eq = np.mean(H[N_PILOT_SYM:, :, :, :], axis=0)
        H_for_eq = np.transpose(H_for_eq, (1, 0, 2))
    else:
        H_for_eq = H_est

    s_hat = mmse_equalize(y_bs, H_for_eq, noise_var, N_PILOT_SYM)
    bits_hat = qpsk_demod_hard(s_hat)

    bb = bits_hat.reshape(bits_hat.shape[0] * bits_hat.shape[1], order='F')
    bits_tx_interleaved = bits_tx.reshape(2, -1).reshape(-1, order='F')
    return np.mean(bb != bits_tx_interleaved)


def main():
    H, N_sc, N_ofdm_total = load_channel(PKL_PATH, UE_IDX)
    Nsc_pilot = N_sc // COMB_STEP
    N_sym_use = N_PILOT_SYM + N_DATA_SYM
    H = H[:N_sym_use, :, :, :]

    scale_pilot, scale_data = scale_pilot_vs_data(N_sc, Nsc_pilot)
    rank1_p, rank2_p, rank1_s2, rank2_s2 = build_pilot_symbols(N_sc, Nsc_pilot)
    _, pilot_sc_idx = build_pilot_packet(N_sc, Nsc_pilot, rank1_p, rank2_p, rank1_s2, rank2_s2, scale_pilot)

    total_steps = len(CE_MODES) * len(SNR_DB_RANGE) * N_TRIALS
    step = 0
    results = {}
    for ce_mode in CE_MODES:
        use_ice = ce_mode == "ICE"
        ber_list = []
        for snr_idx, snr_db in enumerate(SNR_DB_RANGE):
            bers = []
            for trial_idx in range(N_TRIALS):
                step += 1
                pct = 100.0 * step / total_steps
                print(f"\rCE={ce_mode}  SNR_idx={snr_idx + 1}/{len(SNR_DB_RANGE)}  SNR={snr_db:+.0f} dB  trial={trial_idx + 1}/{N_TRIALS}  ({pct:.1f}%)   ", end="", flush=True)
                ber = run_trial(H, N_sc, Nsc_pilot, pilot_sc_idx, rank1_p, rank2_p, rank1_s2, rank2_s2,
                                scale_pilot, scale_data, snr_db, use_ice)
                bers.append(ber)
            ber_list.append(np.mean(bers))
        results[ce_mode] = ber_list
    print()

    # Plot BER(SNR) for all CE modes (clip zeros for log scale)
    plt.figure(figsize=(8, 5))
    for ce_mode in CE_MODES:
        ber_plot = np.maximum(results[ce_mode], 1e-8)
        plt.semilogy(SNR_DB_RANGE, ber_plot, "o-", label=ce_mode, markersize=4)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER vs SNR (ICE vs RCE)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig("ber_vs_snr.png", dpi=150)
    plt.show()
    print("Saved ber_vs_snr.png")


if __name__ == "__main__":
    main()
