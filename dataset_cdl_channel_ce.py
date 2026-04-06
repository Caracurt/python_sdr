"""
Dataset for Transformer-based channel estimation.
Parses CDL channel pkl: (noisy channel, ideal channel) pairs.
Noisy = ideal H + AWGN at fixed SNR or random SNR range.
With N_PILOT_SYM=1, COMB_STEP=1, RANK=1 this is equivalent to a trivial
LS CE (no pilot matrix inversion, no interpolation, single rank).
"""
import pickle
import numpy as np

PKL_PATH = "cdl_channel_Hfr.pkl"
COMB_STEP = 1
TRAIN_RATIO = 0.8
RANK = 1
N_PILOT_SYM = 1


def load_hfr(pkl_path):
    with open(pkl_path, "rb") as f:
        Hfr = pickle.load(f)
    return Hfr


def add_awgn_at_snr(H, snr_db, rng=None):
    """Add complex AWGN so effective SNR is snr_db (dB). H: (S, 256) complex."""
    if rng is None:
        rng = np.random.default_rng()
    signal_power = np.mean(np.abs(H) ** 2)
    noise_var = signal_power / (10 ** (snr_db / 10))
    std = np.sqrt(noise_var / 2)
    noise = std * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
    return (H + noise).astype(np.complex64), noise.astype(np.complex64)


def build_sample_indices(Hfr):
    """List of (ofdm_idx, ue_idx, ue_ant_idx) for each sample."""
    N_ofdm, N_ue, N_ue_ant, _, _ = Hfr.shape
    indices = []
    for ofdm_idx in range(N_ofdm):
        for ue_idx in range(N_ue):
            for ue_ant_idx in range(N_ue_ant):
                indices.append((ofdm_idx, ue_idx, ue_ant_idx))
    return indices


class CDLChannelDataset:
    """
    PyTorch-style dataset: (input, target) where both are (N_sc, N_bs_ant) complex.
    input = ideal H + AWGN at SNR; target = ideal H.
    """

    def __init__(self, pkl_path=PKL_PATH, snr_db=10.0, use_comb2=False,
                 train=True, train_ratio=TRAIN_RATIO, seed=42):
        # snr_db can be:
        # - float/int: fixed SNR
        # - tuple/list(len=2): uniform random SNR in [min, max]
        if isinstance(snr_db, (tuple, list, np.ndarray)):
            if len(snr_db) != 2:
                raise ValueError("snr_db range must have exactly 2 values: (min_db, max_db)")
            self.snr_db_min = float(min(snr_db[0], snr_db[1]))
            self.snr_db_max = float(max(snr_db[0], snr_db[1]))
            self.snr_db = None
        else:
            self.snr_db = float(snr_db)
            self.snr_db_min = None
            self.snr_db_max = None
        self.use_comb2 = use_comb2
        self.train = train
        Hfr = load_hfr(pkl_path)
        self.Hfr = Hfr
        N_ofdm, N_ue, N_ue_ant, N_bs_ant, N_sc = Hfr.shape
        self.N_sc = N_sc
        self.N_bs_ant = N_bs_ant
        indices = build_sample_indices(Hfr)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(indices))
        n_train = int(len(indices) * train_ratio)
        if train:
            self.indices = [indices[i] for i in perm[:n_train]]
        else:
            self.indices = [indices[i] for i in perm[n_train:]]
        self.rng = np.random.default_rng(seed + (0 if train else 1))

    def __len__(self):
        return len(self.indices)

    def _get_H(self, ofdm_idx, ue_idx, ue_ant_idx):
        H = self.Hfr[ofdm_idx, ue_idx, ue_ant_idx, :, :].T.copy()
        return H

    def _sample_snr_db(self):
        if self.snr_db is not None:
            return self.snr_db
        return float(self.rng.uniform(self.snr_db_min, self.snr_db_max))

    def __getitem__(self, idx):
        ofdm_idx, ue_idx, ue_ant_idx = self.indices[idx]
        H_ideal = self._get_H(ofdm_idx, ue_idx, ue_ant_idx)
        if self.use_comb2:
            H_ideal = H_ideal[::COMB_STEP, :]
        snr_db_now = self._sample_snr_db()
        H_noisy, _ = add_awgn_at_snr(H_ideal, snr_db_now, self.rng)
        return H_noisy.astype(np.complex64), H_ideal.astype(np.complex64)


def complex_to_real_512(H):
    """(..., 256) complex -> (..., 512) real (split real and imaginary)."""
    H = np.asarray(H, dtype=np.complex64)
    real = np.real(H)
    imag = np.imag(H)
    return np.concatenate([real, imag], axis=-1).astype(np.float32)


def real_512_to_complex(x):
    """(..., 512) real -> (..., 256) complex."""
    x = np.asarray(x, dtype=np.float32)
    half = x.shape[-1] // 2
    return (x[..., :half] + 1j * x[..., half:]).astype(np.complex64)
