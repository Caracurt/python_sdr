"""
Uplink rank-2 wireless link simulator (frequency-domain).
One UE from CDL pkl, 2 antennas, rank-2 transmission to 256-antenna BS.
2 pilot OFDM symbols (even subcarriers, rank-separated) + N data OFDM symbols.
Modulation: selectable via MODULATION global (QPSK / QAM16 / QAM64 / QAM256).
Channel applied in frequency domain; LS CE with IFFT/window/FFT; MMSE eq; BER (averaged over ranks).
CE modes: ICE (ideal), RCE (real/conventional), DNN (conventional + Transformer denoiser), CS (compressed sensing in beam domain, AMP).
"""
import pickle
import numpy as np
import tensorflow as tf
import sionna as sn
import matplotlib.pyplot as plt

try:
    import torch
    from train_transformer_ce import TransformerCEDenoiser
    _DNN_AVAILABLE = True
except ImportError:
    torch = None
    TransformerCEDenoiser = None
    _DNN_AVAILABLE = False

# --- Config ---
PKL_PATH = "cdl_channel_Hfr.pkl"
UE_IDX = 0
N_PILOT_SYM = 2
N_DATA_SYM = 10
RANK = 2
COMB_STEP = 2  # pilot on even subcarriers only
CE_WINDOW_TAPS = 8    # rectangular window for RCE: keep first W taps
CE_WIN_LEFT = 4       # keep last WinLeft taps for RCE

# --- Modulation ---
MODULATION = "QAM256"   # "QPSK", "QAM16", "QAM64", or "QAM256"
MOD_TO_BITS = {"QPSK": 2, "QAM16": 4, "QAM64": 6, "QAM256": 8}
NUM_BITS_PER_SYM = MOD_TO_BITS[MODULATION]

# BER vs SNR sweep: SNR from -25 to -12 dB, step 1
#SNR_DB_RANGE = np.arange(-25, -10, 2)

#SNR_DB_RANGE = [-20, -18.5, -17, -15.5, -14]
SNR_DB_RANGE = [0, 5, 10]

#CE_MODES = ["ICE", "RCE", "DNN"]
CE_MODES = ["RCE", "DNN", "ICE"]
N_TRIALS = 10  # trials per (SNR, mode) for averaging BER
DNN_CE_MODEL_PATH = "transformer_ce_best_lcx.pt"  # trained Transformer .pt for DNN CE mode # transformer_ce_best_lc.pt transformer_ce_best_lc1.pt
# transformer_ce_best_lc2.pt
USE_PILOT_DATA_SCALING = True  # if False, scale_pilot=1.0 and scale_data=1.0
# CS (compressed sensing) mode: beam-domain sparsity + AMP
CS_N_BEAMS = 12          # number of tracked spatial components (sparsity level K)
CS_AMP_ITERS = 20        # AMP algorithm iterations


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
    constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYM, trainable=False)
    mapper = sn.mapping.Mapper(constellation=constellation)
    num_sym = N_DATA_SYM * N_sc * RANK
    num_bits = num_sym * NUM_BITS_PER_SYM
    bits = np.random.randint(0, 2, (1, num_bits)).astype(np.float32)
    sym = mapper(tf.constant(bits)).numpy().squeeze()
    sym = np.reshape(sym, (N_DATA_SYM, N_sc, RANK), order='F')
    sym = np.transpose(sym, (2, 1, 0))
    return sym * scale_data, bits.squeeze()


def scale_pilot_vs_data(N_sc, Nsc_pilot):
    if not USE_PILOT_DATA_SCALING:
        return 1.0, 1.0
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


def ce_interpolate(H_ls, pilot_sc_idx, N_sc, Nsc_pilot,
                    window_taps=None, win_left=None):
    if window_taps is None:
        window_taps = CE_WINDOW_TAPS
    if win_left is None:
        win_left = CE_WIN_LEFT
    N_fft_ce = int(2 ** np.ceil(np.log2(Nsc_pilot)))
    W = min(window_taps, N_fft_ce // 2)
    WinLeft = min(win_left, N_fft_ce // 2)
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


def load_dnn_ce_model(pt_path, device=None):
    """Load trained Transformer CE model from .pt. Returns (model, device) or (None, None) if unavailable."""
    if not _DNN_AVAILABLE or torch is None:
        return None, None
    try:
        try:
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(pt_path, map_location="cpu")
    except Exception:
        return None, None
    config = ckpt.get("config", {})
    model = TransformerCEDenoiser(
        N_sc=config.get("N_sc", 256),
        N_bs_ant=config.get("N_bs_ant", 256),
        Nl=config.get("Nl", 12),
        Nr=config.get("Nr", 36),
        d_model=config.get("d_model", 16),
        n_head=config.get("n_head", 1),
        n_layers=config.get("n_layers", 6),
        conv_kernel=config.get("conv_kernel", 3),
        conv_padding=config.get("conv_padding", 1),
        n_fft_pre=config.get("n_fft_pre", None),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device


def make_beam_matrix(N):
    """Orthonormal DFT matrix for antenna-to-beam decomposition (N x N)."""
    return np.fft.fft(np.eye(N, dtype=np.complex64), axis=0, norm="ortho")


def amp_denoise_one(h_est, B, K, n_iters):
    """
    AMP algorithm: estimate sparse beam-domain vector x from y = B @ x + noise.
    h_est: (N,) complex (RCE output for one subcarrier, one UE ant).
    B: (N, N) beam matrix. Returns h_denoised (N,) complex.
    """
    N = B.shape[0]
    K = min(K, N)
    y = np.asarray(h_est, dtype=np.complex128)
    z = y.copy()
    x_hat = np.zeros(N, dtype=np.complex128)
    for _ in range(n_iters):
        r = B.conj().T @ z + x_hat
        idx = np.argsort(np.abs(r))[::-1]
        x_hat = np.zeros(N, dtype=np.complex128)
        x_hat[idx[:K]] = r[idx[:K]]
        z = y - B @ x_hat + (K / N) * z
    return (B @ x_hat).astype(np.complex64)


def apply_cs_ce(H_est, B, K, n_iters):
    """Apply CS (beam-domain AMP) denoising to RCE output. H_est: (256, RANK, N_sc) -> (256, RANK, N_sc)."""
    N_bs, RANK, N_sc = H_est.shape
    out = np.zeros_like(H_est)
    for r in range(RANK):
        for k in range(N_sc):
            out[:, r, k] = amp_denoise_one(H_est[:, r, k], B, K, n_iters)
    return out


def apply_dnn_ce(H_est, model, device):
    """Apply DNN (Transformer) denoiser to conventional CE output. H_est: (256, RANK, N_sc) complex -> (256, RANK, N_sc) complex."""
    if not _DNN_AVAILABLE or model is None:
        return H_est
    N_bs, RANK, N_sc = H_est.shape
    out = np.zeros_like(H_est)
    for r in range(RANK):
        h = H_est[:, r, :].T.astype(np.complex64)      # (N_sc, N_bs_ant)
        h_t = torch.from_numpy(h).unsqueeze(0).to(device)  # (1, N_sc, N_bs_ant) complex
        with torch.no_grad():
            h_out_t = model(h_t)                         # (1, N_sc, N_bs_ant) complex
        h_out = h_out_t.cpu().numpy().squeeze(0)         # (N_sc, N_bs_ant)
        out[:, r, :] = h_out.T
    return out.astype(np.complex64)


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


def demod_hard(s_hat, scale_data):
    """Hard-decision demapping for the selected modulation (QPSK/QAM16/QAM64/QAM256).
    s_hat: (RANK, N_sc, N_data) complex equalized symbols (still scaled by scale_data).
    Returns: list of length RANK, each element is 1-D bit array for that rank,
    ordered with time varying fastest then subcarrier (matching TX F-order reshape).
    """
    constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYM, trainable=False)
    demapper = sn.mapping.Demapper("maxlog", constellation=constellation)
    bits_per_rank = []
    for r in range(RANK):
        # s_hat[r]: (N_sc, N_data) → flatten so time varies fastest, then sc (matches TX F-order)
        sr = s_hat[r].T.flatten(order='F')
        sr = sr / scale_data                          # undo data scaling to match constellation
        sr_tf = tf.constant(sr.reshape(-1, 1))        # (N_sym, 1) for demapper
        llr = demapper([sr_tf, tf.constant(1.0)]).numpy()  # (N_sym, NUM_BITS_PER_SYM)
        bits_per_rank.append((llr > 0).astype(np.float32).flatten())
    return bits_per_rank


def _add_awgn_to_channel(H, snr_db):
    """Add complex AWGN to channel matrix at given SNR. Matches training dataset."""
    sig_pow = np.mean(np.abs(H) ** 2)
    noise_var = sig_pow / (10 ** (snr_db / 10))
    std = np.sqrt(noise_var / 2)
    noise = std * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
    return (H + noise).astype(np.complex64)


def compute_nmse(H_est, H_ideal):
    """NMSE = sum(|H_est - H_ideal|^2) / sum(|H_ideal|^2)."""
    diff = H_est - H_ideal
    num = np.sum(np.abs(diff) ** 2)
    den = np.sum(np.abs(H_ideal) ** 2)
    if den < 1e-30:
        return 0.0
    return float(num / den)


def run_trial(H, N_sc, Nsc_pilot, pilot_sc_idx, rank1_p, rank2_p, rank1_s2, rank2_s2,
              scale_pilot, scale_data, snr_db, ce_mode, dnn_model=None, dnn_device=None, cs_B=None):
    """Run one trial: TX, channel, AWGN, CE, equalize, demod. Returns (BER, NMSE_CE)."""
    data_sym, bits_tx = build_data_packet(N_sc, scale_data)
    x, _ = build_pilot_packet(N_sc, Nsc_pilot, rank1_p, rank2_p, rank1_s2, rank2_s2, scale_pilot)
    x[:, :, N_PILOT_SYM:] = data_sym

    y_bs = apply_channel(H, x)
    y_bs, noise_var = add_awgn(y_bs, snr_db)

    H_ls = ls_ce_pilots(y_bs, pilot_sc_idx, rank1_p, rank2_p, rank1_s2, rank2_s2, scale_pilot)

    H_ideal = np.mean(H[N_PILOT_SYM:, :, :, :], axis=0)
    H_ideal = np.transpose(H_ideal, (1, 0, 2))  # (256, RANK, N_sc)

    if ce_mode == "ICE":
        H_for_eq = H_ideal
    elif ce_mode == "RCE":
        H_est = ce_interpolate(H_ls, pilot_sc_idx, N_sc, Nsc_pilot)
        H_for_eq = H_est
    elif ce_mode == "CS":
        H_est = ce_interpolate(H_ls, pilot_sc_idx, N_sc, Nsc_pilot)
        H_for_eq = apply_cs_ce(H_est, cs_B, CS_N_BEAMS, CS_AMP_ITERS)
    else:
        assert ce_mode == "DNN", ce_mode
        H_noisy_dnn = _add_awgn_to_channel(H_ideal, snr_db)
        H_for_eq = apply_dnn_ce(H_noisy_dnn, dnn_model, dnn_device)

    nmse_ce = compute_nmse(H_for_eq, H_ideal)

    # plot worst chan
    nmse_c_arr = np.abs(H_for_eq[:, 0, :] - H_ideal[:, 0, :])**2
    nmse_c_arr_b = np.mean(nmse_c_arr, axis=1)
    idx_worst = np.argmax(nmse_c_arr_b)
    #print(f'Worst channel: {idx_worst}')
    #print(f'Worst channel NMSE: {nmse_c_arr_b[idx_worst]}')

    if False:
        plt.figure(1)
        plt.plot(np.arange(0, N_sc), np.abs(H_for_eq[:, 0, :][idx_worst]), label='Estimated')
        plt.plot(np.arange(0, N_sc), np.abs(H_ideal[:, 0, :][idx_worst]), label='Ideal')
        plt.legend()
        plt.grid()
        plt.show()


    s_hat = mmse_equalize(y_bs, H_for_eq, noise_var, N_PILOT_SYM)
    bits_hat = demod_hard(s_hat, scale_data)

    # Per-rank BER: bits_tx was produced by F-order reshape (N_DATA_SYM, N_sc, RANK),
    # so bits for rank r occupy a contiguous block in bits_tx.
    bits_per_rank_len = N_DATA_SYM * N_sc * NUM_BITS_PER_SYM
    total_err = 0
    for r in range(RANK):
        bits_tx_r = bits_tx[r * bits_per_rank_len : (r + 1) * bits_per_rank_len]
        total_err += np.sum(bits_hat[r] != bits_tx_r)
    ber = total_err / (RANK * bits_per_rank_len)
    return ber, nmse_ce


def main():
    H, N_sc, N_ofdm_total = load_channel(PKL_PATH, UE_IDX)
    Nsc_pilot = N_sc // COMB_STEP
    N_sym_use = N_PILOT_SYM + N_DATA_SYM
    H = H[:N_sym_use, :, :, :]

    scale_pilot, scale_data = scale_pilot_vs_data(N_sc, Nsc_pilot)
    rank1_p, rank2_p, rank1_s2, rank2_s2 = build_pilot_symbols(N_sc, Nsc_pilot)
    _, pilot_sc_idx = build_pilot_packet(N_sc, Nsc_pilot, rank1_p, rank2_p, rank1_s2, rank2_s2, scale_pilot)

    step = 0
    results = {}
    dnn_model, dnn_device = None, None
    cs_B = None
    ce_modes = list(CE_MODES)
    if "DNN" in ce_modes and _DNN_AVAILABLE:
        dnn_model, dnn_device = load_dnn_ce_model(DNN_CE_MODEL_PATH)
        if dnn_model is None:
            print("Warning: DNN CE requested but model not loaded; skipping DNN mode.")
            ce_modes = [m for m in ce_modes if m != "DNN"]
    if "CS" in ce_modes:
        cs_B = make_beam_matrix(256)
    total_steps = len(ce_modes) * len(SNR_DB_RANGE) * N_TRIALS

    results_nmse = {}

    for ce_mode in ce_modes:
        ber_list = []
        nmse_list = []
        for snr_idx, snr_db in enumerate(SNR_DB_RANGE):
            bers = []
            nmses = []
            for trial_idx in range(N_TRIALS):
                step += 1
                pct = 100.0 * step / total_steps
                print(f"\rCE={ce_mode}  SNR_idx={snr_idx + 1}/{len(SNR_DB_RANGE)}  SNR={snr_db:+.0f} dB  trial={trial_idx + 1}/{N_TRIALS}  ({pct:.1f}%)   ", end="", flush=True)
                ber, nmse_ce = run_trial(H, N_sc, Nsc_pilot, pilot_sc_idx, rank1_p, rank2_p, rank1_s2, rank2_s2,
                                         scale_pilot, scale_data, snr_db, ce_mode, dnn_model, dnn_device, cs_B)
                bers.append(ber)
                nmses.append(nmse_ce)
            avg_ber = np.mean(bers)
            avg_nmse = np.mean(nmses)
            ber_list.append(avg_ber)
            nmse_list.append(avg_nmse)
            print(f"  -> avg BER={avg_ber:.6f}  avg NMSE={avg_nmse:.6f}")
        results[ce_mode] = ber_list
        results_nmse[ce_mode] = nmse_list
    print()

    fig, (ax_ber, ax_nmse) = plt.subplots(1, 2, figsize=(14, 5))

    for ce_mode in results:
        ber_plot = np.maximum(results[ce_mode], 1e-8)
        ax_ber.semilogy(SNR_DB_RANGE, ber_plot, "o-", label=ce_mode, markersize=4)
    ax_ber.set_xlabel("SNR (dB)")
    ax_ber.set_ylabel("BER")
    ax_ber.set_title("BER vs SNR")
    ax_ber.legend()
    ax_ber.grid(True, which="both", ls="-", alpha=0.5)

    for ce_mode in results_nmse:
        if ce_mode == "ICE":
            continue
        nmse_plot = np.maximum(results_nmse[ce_mode], 1e-8)
        ax_nmse.semilogy(SNR_DB_RANGE, nmse_plot, "s-", label=ce_mode, markersize=4)
    ax_nmse.set_xlabel("SNR (dB)")
    ax_nmse.set_ylabel("NMSE")
    ax_nmse.set_title("CE NMSE vs SNR")
    ax_nmse.legend()
    ax_nmse.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig("ber_nmse_vs_snr.png", dpi=150)
    plt.show()
    print("Saved ber_nmse_vs_snr.png")


if __name__ == "__main__":
    main()
