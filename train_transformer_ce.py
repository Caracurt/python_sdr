"""
Train Transformer-based channel estimation with beam-time preprocessing.
Preprocessing (non-trainable): beam DFT -> IFFT -> truncate (last Nl + first Nr) -> IQ split.
Model: Conv1D embedding (2->16) + 6 Transformer layers (d_model=16, n_head=1) + Conv1D output (16->2).
Postprocessing (non-trainable): IQ merge -> zero-pad -> FFT -> inverse beam DFT.
Loss: NMSE in original frequency-antenna domain. Optimizer: Adam. Early stopping.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_cdl_channel_ce import CDLChannelDataset

# --- Config ---
PKL_PATH = "cdl_channel_Hfr_2x1.pkl"
#PKL_PATH = "cdl_channel_Hfr_new.pkl"
SNR_DB_MIN = 0.0
SNR_DB_MAX = 30.0
USE_COMB2 = False
TRAIN_RATIO = 0.8
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 500
EARLY_STOP_PATIENCE = 30
EARLY_STOP_TOL = 1e-6
D_MODEL = 16
N_HEAD = 1
N_LAYERS = 4
CONV_KERNEL = 3
CONV_PADDING = 1
NL = 8   # last time-domain taps to keep # default 12
NR = 16   # first time-domain taps to keep # default 36
AUG_TAU_MAX = 0.1   # data augmentation: random time-sync error in [-AUG_TAU_MAX, AUG_TAU_MAX]
SEED = 42


def nmse_loss(H_est, H_ideal):
    """NMSE on complex tensors: sum(|est-ideal|^2) / sum(|ideal|^2)."""
    diff = H_est - H_ideal
    num = torch.sum(diff.real ** 2 + diff.imag ** 2)
    den = torch.sum(H_ideal.real ** 2 + H_ideal.imag ** 2)
    return num / (den + 1e-12)


def snr_weighted_nmse_loss(H_est, H_ideal, H_noisy):
    """SNR-weighted NMSE to balance mixed-SNR training batches.

    Weight per sample = linear SNR estimated from ideal/noisy pair:
      snr_lin = sum(|H_ideal|^2) / sum(|H_noisy - H_ideal|^2)
    Loss per sample = snr_lin * NMSE_sample
    """
    err = H_est - H_ideal
    noise = H_noisy - H_ideal

    # Per-sample powers over subcarrier/antenna axes
    p_err = torch.sum(err.real ** 2 + err.imag ** 2, dim=(1, 2))
    p_sig = torch.sum(H_ideal.real ** 2 + H_ideal.imag ** 2, dim=(1, 2))
    p_noise = torch.sum(noise.real ** 2 + noise.imag ** 2, dim=(1, 2))

    nmse_per = p_err / (p_sig + 1e-12)
    snr_lin_per = p_sig / (p_noise + 1e-12)
    loss_per = snr_lin_per * nmse_per
    return torch.mean(loss_per)


def time_sync_augment(noisy, ideal, tau_max=AUG_TAU_MAX):
    """Apply random time-sync error: multiply both tensors by exp(j*tau*k).
    noisy, ideal: (B, N_sc, N_bs_ant) complex.  tau ~ Uniform[-tau_max, tau_max] per sample."""
    B, N_sc, _ = noisy.shape
    tau = (2.0 * tau_max) * torch.rand(B, device=noisy.device) - tau_max   # (B,)
    k = torch.arange(N_sc, device=noisy.device, dtype=torch.float32)       # (N_sc,)
    phase = tau.unsqueeze(1) * k.unsqueeze(0)                              # (B, N_sc)
    comp = torch.polar(torch.ones_like(phase), phase).unsqueeze(2)         # (B, N_sc, 1)
    return noisy * comp, ideal * comp


class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer layer with residuals around both sub-blocks.
    LN -> MHA -> +residual -> LN -> Conv1D -> GELU -> +residual.
    """

    def __init__(self, d_model, n_head, conv_kernel=3, conv_padding=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, conv_kernel, padding=conv_padding)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B*N_beam, seq_len=48, d_model)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm2 = self.norm2(x)
        x_c = x_norm2.transpose(1, 2)  # (B*N_beam, d_model, seq_len) for Conv1d
        x_c = self.act(self.conv(x_c))
        x_c = x_c.transpose(1, 2)
        x = x + x_c
        return x


class TransformerCEDenoiser(nn.Module):
    def __init__(self, N_sc, N_bs_ant=256, Nl=12, Nr=36, d_model=16, n_head=1,
                 n_layers=6, conv_kernel=3, conv_padding=1, n_fft_pre=None):
        super().__init__()
        self.N_sc = N_sc
        self.N_bs_ant = N_bs_ant
        self.Nl = Nl
        self.Nr = Nr
        self.d_model = d_model
        if n_fft_pre is None:
            n_fft_pre = 1
            while n_fft_pre <= N_sc:
                n_fft_pre *= 2
        self.N_fft_pre = int(n_fft_pre)
        if self.N_fft_pre <= self.N_sc:
            raise ValueError(f"n_fft_pre must be > N_sc, got n_fft_pre={self.N_fft_pre}, N_sc={self.N_sc}")
        if (self.N_fft_pre & (self.N_fft_pre - 1)) != 0:
            raise ValueError(f"n_fft_pre must be a power of 2, got {self.N_fft_pre}")
        if (self.Nl + self.Nr) > self.N_fft_pre:
            raise ValueError(f"Nl+Nr must be <= n_fft_pre, got {self.Nl + self.Nr} > {self.N_fft_pre}")
        self.n_time = Nl + Nr

        self.embed_conv = nn.Conv1d(2, d_model, conv_kernel, padding=conv_padding)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_head, conv_kernel, conv_padding)
            for _ in range(n_layers)
        ])
        self.output_conv = nn.Conv1d(d_model, 2, conv_kernel, padding=conv_padding)

    # --- Non-trainable time-alignment (matches rx_funcs_mimo CE_mode=5) ---

    def _estimate_phase_slope(self, H):
        """Estimate average phase slope across subcarriers (time delay).
        H: (B, N_sc, N_bs_ant) complex.  Returns avg_angle: (B,) float."""
        h_first = H[:, :-1, :]
        h_second = H[:, 1:, :]
        corr_per_ant = torch.sum(h_first.conj() * h_second, dim=1)  # (B, N_bs_ant)
        avg_angle = torch.mean(torch.angle(corr_per_ant), dim=1)    # (B,)
        return avg_angle

    def _apply_time_alignment(self, H, avg_angle):
        """Remove phase slope → shift channel to zero tap in time domain."""
        k = torch.arange(H.shape[1], device=H.device, dtype=torch.float32)
        phase = -avg_angle.unsqueeze(1) * k.unsqueeze(0)            # (B, N_sc)
        comp = torch.polar(torch.ones_like(phase), phase).unsqueeze(2)
        return H * comp

    def _remove_time_alignment(self, H, avg_angle):
        """Restore original phase slope (inverse of _apply_time_alignment)."""
        k = torch.arange(H.shape[1], device=H.device, dtype=torch.float32)
        phase = avg_angle.unsqueeze(1) * k.unsqueeze(0)
        comp = torch.polar(torch.ones_like(phase), phase).unsqueeze(2)
        return H * comp

    # --- Beam-time preprocessing / postprocessing ---

    def preprocess(self, H):
        """(B, N_sc, N_bs_ant) complex -> (B, N_beam, 2, n_time) float."""
        H_beam = torch.fft.fft(H, dim=-1, norm='ortho')
        # Use larger FFT size in delay domain (power-of-2 and strictly > N_sc), e.g. 64 -> 128.
        h_time = torch.fft.ifft(H_beam, n=self.N_fft_pre, dim=-2, norm='ortho')
        h_last = h_time[:, -self.Nl:, :]
        h_first = h_time[:, :self.Nr, :]
        h_trunc = torch.cat([h_last, h_first], dim=1)  # (B, Nl+Nr, N_beam)
        h_real = h_trunc.real
        h_imag = h_trunc.imag
        h_iq = torch.stack([h_real, h_imag], dim=-1)   # (B, n_time, N_beam, 2)
        return h_iq.permute(0, 2, 3, 1)                # (B, N_beam, 2, n_time)

    def postprocess(self, x_iq):
        """(B, N_beam, 2, n_time) float -> (B, N_sc, N_bs_ant) complex."""
        x_iq = x_iq.permute(0, 3, 1, 2)                # (B, n_time, N_beam, 2)
        h_complex = torch.complex(x_iq[..., 0], x_iq[..., 1])  # (B, n_time, N_beam)
        B = h_complex.shape[0]
        h_first = h_complex[:, self.Nl:, :]             # (B, Nr, N_beam)
        h_last = h_complex[:, :self.Nl, :]              # (B, Nl, N_beam)
        mid_len = self.N_fft_pre - self.Nr - self.Nl
        zeros_mid = torch.zeros(B, mid_len, self.N_bs_ant,
                                dtype=h_complex.dtype, device=h_complex.device)
        h_padded = torch.cat([h_first, zeros_mid, h_last], dim=1)  # (B, N_fft_pre, N_beam)
        H_beam_full = torch.fft.fft(h_padded, n=self.N_fft_pre, dim=-2, norm='ortho')
        H_beam = H_beam_full[:, :self.N_sc, :]  # Crop back to original subcarrier count
        H_ant = torch.fft.ifft(H_beam, dim=-1, norm='ortho')
        return H_ant

    def forward(self, H_noisy):
        """(B, N_sc, N_bs_ant) complex -> (B, N_sc, N_bs_ant) complex."""
        B = H_noisy.shape[0]

        # Non-trainable: estimate and remove phase slope (time-align to zero tap)
        avg_angle = self._estimate_phase_slope(H_noisy)
        H_aligned = self._apply_time_alignment(H_noisy, avg_angle)

        x = self.preprocess(H_aligned)                    # (B, N_beam, 2, n_time)
        x = x.reshape(B * self.N_bs_ant, 2, self.n_time)  # (B*N_beam, 2, n_time)
        x = self.embed_conv(x)                             # (B*N_beam, d_model, n_time)
        x = x.transpose(1, 2)                              # (B*N_beam, n_time, d_model)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)                              # (B*N_beam, d_model, n_time)
        x = self.output_conv(x)                             # (B*N_beam, 2, n_time)
        x = x.reshape(B, self.N_bs_ant, 2, self.n_time)    # (B, N_beam, 2, n_time)
        H_out = self.postprocess(x)

        # Non-trainable: restore original phase slope
        return self._remove_time_alignment(H_out, avg_angle)


def collate_complex(batch):
    noisy_list, ideal_list = zip(*batch)
    noisy = np.stack(noisy_list, axis=0)
    ideal = np.stack(ideal_list, axis=0)
    return torch.from_numpy(noisy), torch.from_numpy(ideal)


def main():
    parser = argparse.ArgumentParser(description="Train Transformer CE with beam-time preprocessing")
    parser.add_argument("--pkl", default=PKL_PATH, help="Path to CDL channel pkl")
    parser.add_argument("--snr_db_min", type=float, default=SNR_DB_MIN, help="Min SNR (dB) for training data")
    parser.add_argument("--snr_db_max", type=float, default=SNR_DB_MAX, help="Max SNR (dB) for training data")
    parser.add_argument("--val_snr_db", type=float, default=None,
                        help="Optional fixed validation SNR (dB). If not set, uses same range as training.")
    parser.add_argument("--comb2", action="store_true", help="Use every second subcarrier only")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--tol", type=float, default=EARLY_STOP_TOL)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    snr_min = float(min(args.snr_db_min, args.snr_db_max))
    snr_max = float(max(args.snr_db_min, args.snr_db_max))
    train_snr = (snr_min, snr_max)
    val_snr = args.val_snr_db if args.val_snr_db is not None else (snr_min, snr_max)

    train_ds = CDLChannelDataset(
        pkl_path=args.pkl, snr_db=train_snr, use_comb2=args.comb2,
        train=True, train_ratio=args.train_ratio, seed=args.seed,
    )
    val_ds = CDLChannelDataset(
        pkl_path=args.pkl, snr_db=val_snr, use_comb2=args.comb2,
        train=False, train_ratio=args.train_ratio, seed=args.seed,
    )
    N_sc = train_ds[0][0].shape[0]
    N_bs_ant = train_ds[0][0].shape[1]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_complex)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_complex)

    model = TransformerCEDenoiser(
        N_sc=N_sc, N_bs_ant=N_bs_ant, Nl=NL, Nr=NR,
        d_model=D_MODEL, n_head=N_HEAD, n_layers=N_LAYERS,
        conv_kernel=CONV_KERNEL, conv_padding=CONV_PADDING,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for noisy, ideal in train_loader:
            noisy = noisy.to(device)
            ideal = ideal.to(device)
            noisy, ideal = time_sync_augment(noisy, ideal)
            optimizer.zero_grad()
            out = model(noisy)
            loss = snr_weighted_nmse_loss(out, ideal, noisy)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for noisy, ideal in val_loader:
                noisy = noisy.to(device)
                ideal = ideal.to(device)
                out = model(noisy)
                loss = nmse_loss(out, ideal)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        print(f"Epoch {epoch + 1}/{args.epochs}  train NMSE: {train_loss:.6f}  val NMSE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            if best_val_loss - val_loss >= args.tol:
                patience_counter = 0
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs (tolerance {args.tol})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    pkl_stem = Path(args.pkl).stem
    out_pt = f"transformer_ce_best_{pkl_stem}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "N_sc": N_sc, "N_bs_ant": N_bs_ant, "Nl": NL, "Nr": NR,
            "d_model": D_MODEL, "n_head": N_HEAD, "n_layers": N_LAYERS,
            "conv_kernel": CONV_KERNEL, "conv_padding": CONV_PADDING,
            "n_fft_pre": model.N_fft_pre,
            "use_comb2": args.comb2,
            "snr_db_min": snr_min, "snr_db_max": snr_max,
            "val_snr_db": args.val_snr_db,
        },
    }, out_pt)
    print(f"Saved {out_pt}")


if __name__ == "__main__":
    main()
