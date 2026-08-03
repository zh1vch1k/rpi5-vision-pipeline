#pragma once

#include <iostream>
#include <fcntl.h>        
#include <sys/mman.h>    
#include <semaphore.h>   
#include <unistd.h> 
#include <cstdint>    

#include <opencv2/opencv.hpp>
#include "frame_metadata.hpp"
#include "config.hpp"

inline const config::Config ctx = config::getContext("config.json");

class ShmReader { 
private: 
    const char* BUFFER_PATH;
    const char* SEM_NAME;
    size_t TOTAL_BYTES;

    void* mmap_ptr = nullptr;
    sem_t* sem = nullptr;

public:
    ShmReader() = default;

    int init();
    cv::Mat readFrame();
    
    ~ShmReader();
};