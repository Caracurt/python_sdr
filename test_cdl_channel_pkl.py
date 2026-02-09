"""
Simple test: read CDL channel from pkl and verify shape.
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

PKL_FILE = "cdl_channel_Hfr.pkl"



def main():
    with open(PKL_FILE, "rb") as f:
        Hfr = pickle.load(f)

    # check shape of H
    N_ofdm, N_ue, N_ue_ant, N_bs_ant, N_sc = Hfr.shape

    h_fr_test = Hfr[0, 0, 0, 0, :]
    h_time_test = Hfr[:, 0, 0, 0, 0]

    h_ant_test = Hfr[0, 0, 0, :, :] # 256 x 192
    h_ant_test = h_ant_test.transpose(1, 0) # 192 x 256

    h_3d = h_ant_test.reshape(-1, 8, 16, 2, order='C')

    hp1 = h_3d[:, :, :, 0] # 8x16
    hp2 = h_3d[:, :, :, 1] # 8x16

    DFT1 = np.fft.fft(np.eye(16, dtype=np.complex64), axis=0)
    DFT2 = np.fft.fft(np.eye(8, dtype=np.complex64), axis=0)

    h_p1 = hp1 @ DFT1.T
    h_p1 = DFT2 @ h_p1

    plt.imshow(np.mean(np.abs(h_p1)**2, axis=0), aspect='auto')
    plt.colorbar()
    plt.show()

    h_p2 = hp2 @ DFT1.T
    h_p2 = DFT2 @ h_p2

    plt.imshow(np.mean(np.abs(h_p2)**2, axis=0), aspect='auto')
    plt.colorbar()
    plt.show()


    plt.figure(1)
    plt.plot(np.arange(0, N_sc), np.real(h_fr_test))
    plt.xlabel('Subcarrier index')
    plt.ylabel('Real part of channel frequency response')
    plt.grid()
    plt.figure(2)
    plt.plot(np.arange(0, N_ofdm), np.real(h_time_test))
    plt.xlabel('OFDM symbol index')
    plt.grid()
    plt.show()




if __name__ == "__main__":
    main()
