import struct
import posix_ipc
from multiprocessing.shared_memory import SharedMemory



BUFFER_PATH: str = 'rpi5_pipeline'

#3 channels for image 640*640 in bytes + 32 bits of metadata
FRAME_SIZE = 640 * 640 * 3  
HEADER_SIZE = 32            
TOTAL_BYTES = HEADER_SIZE + FRAME_SIZE

shm = SharedMemory(name=BUFFER_PATH, create=True, size=TOTAL_BYTES)

def write_frame(BUFFER_PATH, frame_id, timestamp_ns, frame):
    height = frame.shape[0]
    width = frame.shape[1]
    channels = frame.shape[2]
    reserved = 0

    frame_metadata = struct.pack('=QIIIQI', frame_id, width, height, channels, timestamp_ns, reserved)

    shm.buf[0: 32] = frame_metadata
    shm.buf[32: TOTAL_BYTES] = memoryview(frame)


