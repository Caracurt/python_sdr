import numpy as np
import re

class SysParUL:
    def __init__(self, inPar_dict):
        self.sample_rate = inPar_dict["sample_rate"]
        self.center_freq = inPar_dict["center_freq"]
        self.device_type = inPar_dict["device_type"]
        self.Ntx = inPar_dict["Ntx"]
        self.delta_f = inPar_dict["delta_f"]
        self.frac_guard = inPar_dict["frac_guard"]
        self.do_cfo_corr = inPar_dict["do_cfo_corr"]
        self.do_ce = inPar_dict["do_ce"]
        self.dc_offset = inPar_dict["dc_offset"]
        self.pilot_repeat = inPar_dict["pilot_repeat"]
        self.do_sc_fdm = inPar_dict["do_sc_fdm"]
        self.do_fec = inPar_dict["do_fec"]
        self.CR = inPar_dict["CR"]


        self.dummyTx = inPar_dict["dummyTx"]
        self.dummyRx = inPar_dict["dummyRx"]
        self.SNR_dummy = inPar_dict["SNR_dummy"] # SNR for dummy Rx


        self.Ndata = inPar_dict["Ndata"]

        self.delta_t = 1 / self.sample_rate
        self.N_sc_av = int(self.sample_rate / self.delta_f)
        self.N_fft = 2 ** (int(np.log2(self.N_sc_av)))
        self.N_sc_use = int(self.N_fft * self.frac_guard)
        self.guard_length = int(0.5 * self.N_fft)
        self.CP_len = int(self.N_fft * 0.2)
        self.num_bits_sym = 2
        self.BlockSize = self.N_sc_use * self.num_bits_sym

        self.T_prec = np.fft.ifft(np.eye(self.N_sc_use, dtype=np.complex64), norm='ortho')



