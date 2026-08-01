import struct
import numpy as np
import posix_ipc
from multiprocessing.shared_memory import SharedMemory

BUFFER_PATH: str = 'rpi5_pipeline'
SEM_NAME : str = '/rpi5_semaphore'

#3 channels for image 640*640 in bytes + 32 bits of metadata
FRAME_SIZE = 640 * 640 * 3  
HEADER_SIZE = 32            
TOTAL_BYTES = HEADER_SIZE + FRAME_SIZE

shm = SharedMemory(name=BUFFER_PATH, create=True, size=TOTAL_BYTES)
sem = posix_ipc.Semaphore(
    SEM_NAME, flags=posix_ipc.O_CREAT, initial_value=0
)


def write_frame(frame_id: np.ndarray, timestamp_ns: int, frame: int):
    height, width, channels = frame.shape

    frame_metadata = struct.pack('=QIIIQI', frame_id, width, height, channels, timestamp_ns, 0)

    shm.buf[0: HEADER_SIZE] = frame_metadata
    shm.buf[HEADER_SIZE: HEADER_SIZE + frame.nbytes] = memoryview(frame)

    sem.release()


def cleanup():
    shm.close()
    shm.unlink()
    sem.close()
    sem.unlink()
