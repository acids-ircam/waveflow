# RESIDUAL BLOCK HPARAMS
hidden_size = 64
skp_size    = 64
res_size    = 64
kernel_size = 3
cdt_size    = 80
in_size     = 1
out_size    = 2

# FLOW HPARAMS
h           = 16
n_flow      = 8
n_layer     = 8
cycle_size  = 8

class build:
    n_fft = 2048
    n_mel = 80
    wavs = "/fast-2/datasets/Solv4_strings_wav/audio/**/**/*.wav"
    sr = 16000
    hop_length = 128
    
    sequence_size = 2**14 + n_fft - 1
    seq_size_f = 128
    eps = 1e-5