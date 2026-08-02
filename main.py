import time
import lucas_canade
import cv2 as cv
import shm_memory_writer as mem_writer


if __name__ == '__main__':
    frame_id: int = 0

    try:
        for frame in lucas_canade.frame_process():
            mem_writer.write_frame(frame_id, time.time_ns(), frame)
            frame_id += 1
            
            cv.imshow('YOLO + Flow.', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                print("Closing program...")
                break

    except KeyboardInterrupt:
        pass
    finally:
        mem_writer.cleanup()
