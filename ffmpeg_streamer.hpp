#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#include <opencv2/opencv.hpp>

class FfmpegStreamer{
private: 

AVFormatContext* formatContext = nullptr;
AVCodecContext*  codecContext  = nullptr;

AVFrame* rawFrame = nullptr; 
AVFrame* yuvFrame = nullptr;

AVPacket* encodedFrame = nullptr;

SwsContext* swsContext = nullptr;

uint64_t frame_pts {0};


public: 
    FfmpegStreamer(): 
        rawFrame(av_frame_alloc()), 
        yuvFrame(av_frame_alloc()),
        encodedFrame(av_packet_alloc()) {}

    
    int initFormatContext();

    int initCodecContext();

    int initSwsContext();

    int sendFrame(const cv::Mat& wrapper);

    int initStream();
        

    ~FfmpegStreamer() {
        av_frame_free(&yuvFrame);
        av_frame_free(&rawFrame);

        avcodec_free_context(&codecContext);
        avformat_free_context(formatContext);

        sws_freeContext(swsContext);

        av_packet_free(&encodedFrame);

        if (formatContext && !(formatContext->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&formatContext->pb);
        }
    }
};



