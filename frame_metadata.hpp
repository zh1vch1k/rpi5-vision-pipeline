#ifndef __FRAME_METADATA_HPP__
#define __FRAME_METADATA_HPP__
#pragma once

#include <cstdint>

#pragma pack(push, 1)
struct FrameMetadata {
    uint64_t frameId;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint64_t timestampNs;
    uint32_t reserved;
};
#pragma pack(pop)

using Metadata = FrameMetadata;

#endif //__FRAME_METADATA_HPP__