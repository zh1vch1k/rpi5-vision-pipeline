#pragma once

#include <iostream>
#include <fcntl.h>        
#include <sys/mman.h>    
#include <semaphore.h>   
#include <unistd.h> 
#include <cstdint>    

#include <opencv2/opencv.hpp>
#include "frame_metadata.hpp"

class ShmReader { 
private: 
    inline static constexpr const char* BUFFER_PATH = "rpi5_pipeline";
    inline static constexpr const char* SEM_NAME = "/rpi5_semaphore";
    inline static constexpr size_t TOTAL_BYTES = sizeof(FrameMetadata) + (640 * 640 * 3);

    void* mmap_ptr = nullptr;
    sem_t* sem = nullptr;

public:
    ShmReader() = default;
    
    int init();
    cv::Mat readFrame();
    
    ~ShmReader();
};