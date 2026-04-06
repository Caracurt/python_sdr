# Play audio from .mat (FM demodulated): whole-file mode or real-time producer-consumer
import queue
import threading
import numpy as np
import scipy.io
import sounddevice as sd

# Option: True = play whole file first then play (backup); False = producer-consumer (play while loading)
PLAY_WHOLE_FILE = False

SAMPLE_RATE = 48000
BLOCKSIZE = 1024
QUEUE_MAXSIZE = 30  # limit latency in real-time mode

mat = scipy.io.loadmat('fm_record.mat')
H_cell = mat['y_store']
k_num = H_cell.shape[1]


def get_chunk(b_idx):
    """Return one batch as float32 mono (samples, 1)."""
    h = H_cell[0, b_idx][:, 0]
    h = np.asarray(h, dtype=np.float32)
    if h.ndim == 1:
        h = h.reshape(-1, 1)
    return h


if PLAY_WHOLE_FILE:
    # Backup: load entire audio, then play once
    all_chunks = [get_chunk(b_idx) for b_idx in range(k_num)]
    audio_data = np.concatenate(all_chunks, axis=0)
    print(f"Playing whole file: {audio_data.shape[0]} samples")
    sd.play(audio_data, SAMPLE_RATE, blocking=True)
    sd.wait()
    print("Done.")
else:
    # Producer-consumer: play in background while "processing" (iterating) chunks
    audio_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    SENTINEL = object()  # end of stream

    # Buffers for callback: drain one chunk at a time, then take next from queue
    # Use list so callback mutates remainder[0] instead of assigning to 'remainder' (avoids UnboundLocalError)
    remainder = [np.array([]).reshape(0, 1)]
    remainder_lock = threading.Lock()
    stop_requested = [False]  # list so callback can mutate

    def callback(outdata, frames, time_info, status):
        if status:
            print(status)
        with remainder_lock:
            need = frames
            out = np.zeros((frames, 1), dtype=np.float32)
            written = 0
            while written < frames:
                rem = remainder[0]
                if rem.shape[0] >= need:
                    out[written:written + need] = rem[:need]
                    remainder[0] = rem[need:]
                    written += need
                    break
                if rem.shape[0] > 0:
                    n = rem.shape[0]
                    out[written:written + n] = rem
                    written += n
                    need -= n
                    remainder[0] = np.array([]).reshape(0, 1)
                try:
                    chunk = audio_queue.get_nowait()
                except queue.Empty:
                    break
                if chunk is SENTINEL:
                    stop_requested[0] = True
                    break
                rem = np.asarray(chunk, dtype=np.float32)
                if rem.ndim == 1:
                    rem = rem.reshape(-1, 1)
                remainder[0] = rem
            if written < frames:
                out[written:] = 0
            outdata[:] = out
        if stop_requested[0]:
            raise sd.CallbackStop

    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        callback=callback,
        blocksize=BLOCKSIZE,
    ):
        total_queued = 0
        for b_idx in range(k_num):
            chunk = get_chunk(b_idx)
            audio_queue.put(chunk)
            total_queued += chunk.shape[0]
            print(f"Queued chunk b_idx={b_idx} len={chunk.shape[0]}")
        audio_queue.put(SENTINEL)
        # Wait for playback to drain (callback will stop when it sees SENTINEL)
        drain_ms = int(1000 * total_queued / SAMPLE_RATE) + 2000
        sd.sleep(drain_ms)
    print("Done (producer-consumer).")
