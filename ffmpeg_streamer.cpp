#include "ffmpeg_streamer.hpp"

int FfmpegStreamer::initFormatContext() {
    int status = avformat_alloc_output_context2(&formatContext, nullptr, "mpegts", "udp://127.0.0.1:1234");
    if (status < 0) {
        return status;
    }

    if (!(formatContext->oformat->flags & AVFMT_NOFILE)) { 
        if (int status = avio_open2(&formatContext->pb, formatContext->url, AVIO_FLAG_WRITE, nullptr, nullptr) < 0) {
            return status;
        }
    }

    return 0;
}


int FfmpegStreamer::initCodecContext() {
    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);

    codecContext = avcodec_alloc_context3(codec);

    codecContext->width = 1280;
    codecContext->height = 720;
    codecContext->pix_fmt = AV_PIX_FMT_YUV420P;
    codecContext->time_base = {1, 60}; 
    codecContext->max_b_frames = 0;    

    av_opt_set(codecContext->priv_data, "tune", "zerolatency", 0);
    av_opt_set(codecContext->priv_data, "preset", "ultrafast", 0);

    return avcodec_open2(codecContext, codec, nullptr);
}


int FfmpegStreamer::initSwsContext() {
    swsContext = sws_getContext(640, 360, AV_PIX_FMT_BGR24, 
                                1280, 720, AV_PIX_FMT_YUV420P, 
                                SWS_FAST_BILINEAR, 
                                nullptr, nullptr, nullptr);

    yuvFrame->format = AV_PIX_FMT_YUV420P;
    yuvFrame->width  = 1280;
    yuvFrame->height = 720;

    av_frame_get_buffer(yuvFrame, 0);
    return 0;
    }


int FfmpegStreamer::sendFrame(const cv::Mat& wrapper) {
    if (wrapper.empty() || wrapper.type() != CV_8UC3) {
        return -1;
    }

    av_image_fill_arrays(
        rawFrame->data,
        rawFrame->linesize,
        wrapper.data,
        AV_PIX_FMT_BGR24,
        wrapper.cols,
        wrapper.rows,
        1
    );

    int out_height = sws_scale(swsContext, 
                               rawFrame->data,
                               rawFrame->linesize,
                               0,
                               wrapper.rows,
                               yuvFrame->data,
                               yuvFrame->linesize
    );
        
    yuvFrame->pts = frame_pts++;
    if (avcodec_send_frame(codecContext, yuvFrame) >= 0) { 
        while (avcodec_receive_packet(codecContext, encodedFrame) >= 0) {
            av_interleaved_write_frame(formatContext, encodedFrame);
            av_packet_unref(encodedFrame); 
        }
    };
    return 0;
}


int FfmpegStreamer::initStream() {
    AVStream* stream = avformat_new_stream(formatContext, nullptr);

    if (stream == nullptr) {
        return -1;
    }

    avcodec_parameters_from_context(stream->codecpar, codecContext);

    if (avformat_write_header(formatContext, nullptr) < 0) {
        return -1;
    }
    return 0;
}