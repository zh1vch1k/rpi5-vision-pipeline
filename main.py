import time
import lucas_canade
import cv2 as cv
import shm_memory_writer as mem_writer
import queue
import threading

if __name__ == '__main__':
    frame_id: int = 0

    in_queue = queue.Queue(maxsize=1)
    out_queue = queue.Queue(maxsize=1)

    inference_thread = threading.Thread(target = lucas_canade.inference, 
                                        args=(in_queue, out_queue), 
                                        daemon=True)
    inference_thread.start()

    try:
        for frame in lucas_canade.frame_process(in_queue, out_queue):
            mem_writer.write_frame(frame_id, time.time_ns(), frame)
            frame_id += 1
            
            cv.imshow('YOLO + Flow.', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                print("Closing program...")
                break

    except KeyboardInterrupt:
        pass
    finally:
        in_queue.put(None)
        inference_thread.join(timeout=1.0)
        mem_writer.cleanup()
        
