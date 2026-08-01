#include "ffmpeg_streamer.hpp"
#include "shm_memory_reader.hpp"
#include <iostream>
#include <csignal>
#include <atomic>

std::atomic<bool> keep_running{true};

void signal_handler(int signum) {
    std::cout << "\nInterrupt signal (" << signum << ") received. Stopping..." << std::endl;
    keep_running = false;
}

int main(int argc, const char* argv[]) {
    std::signal(SIGINT, signal_handler);

    ShmReader memReader{};
    if (memReader.init() < 0) {
        std::cerr << "Error: Failed to init SHM Reader!" << std::endl;
        return -1;
    }

    FfmpegStreamer streamer{};

    std::cout << "Initializing FFmpeg..." << std::endl;
    if (streamer.initFormatContext() < 0) return -1;
    if (streamer.initCodecContext() < 0) return -1;
    if (streamer.initStream() < 0) return -1; 
    if (streamer.initSwsContext() < 0) return -1;

    std::cout << "Pipeline ready. Waiting for Python frames..." << std::endl;

    try {
        while (keep_running) {
            cv::Mat frame = memReader.readFrame();

            if (frame.empty()) {
                std::cerr << "Warning: Received empty frame from SHM!" << std::endl;
                continue; 
            }

            if (streamer.sendFrame(frame) < 0) {
                std::cerr << "Error: Failed to encode/send frame!" << std::endl;
            }
        }
    } 
    catch (const std::exception& e) {
        std::cerr << "Fatal Exception occurred: " << e.what() << std::endl;
    }

    std::cout << "Shutdown complete." << std::endl;
    return 0;
}