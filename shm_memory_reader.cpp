#include "shm_memory_reader.hpp"

int ShmReader::init() {
    int fd = shm_open(BUFFER_PATH, O_RDONLY, 0666);
    if (fd == -1) {
        std::cerr << "[C++] Error: shm_open hasn't found the path '" << BUFFER_PATH << "'!" << std::endl;
        return -1;
    }

    this->mmap_ptr = mmap(nullptr, TOTAL_BYTES, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);

    if (this->mmap_ptr == MAP_FAILED) {
        std::cerr << "[C++] Error: mmap crashed!" << std::endl;
        this->mmap_ptr = nullptr;
        return -1;
    }

    this->sem = sem_open(SEM_NAME, 0);
    if (this->sem == SEM_FAILED) {
        std::cerr << "[C++] Error: sem_open semaphore hasn't been found '" << SEM_NAME << "'!" << std::endl;
        munmap(this->mmap_ptr, TOTAL_BYTES);
        this->mmap_ptr = nullptr;
        return -1;
    }

    return 0;
}

cv::Mat ShmReader::readFrame() {
    sem_wait(sem);

    auto* meta = static_cast<FrameMetadata*>(mmap_ptr);
    uint8_t* pixels = static_cast<uint8_t*>(mmap_ptr) + sizeof(FrameMetadata);

    return cv::Mat(meta->height, meta->width, CV_8UC3, pixels);
}

ShmReader::~ShmReader() {
    if (mmap_ptr && mmap_ptr != MAP_FAILED) {
        munmap(this->mmap_ptr, TOTAL_BYTES);
        this->mmap_ptr = nullptr;
    }

    if (sem && sem != SEM_FAILED) {
        sem_close(this->sem);
        sem=nullptr;
    }
}