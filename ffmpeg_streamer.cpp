#include "ffmpeg_streamer.hpp"

int FfmpegStreamer::initFormatContext() {
    int status = avformat_alloc_output_context2(&formatContext, nullptr, "mpegts", "udp://192.168.1.118:1221");
    if (status < 0) {
        return status;
    }

    if (!(formatContext->oformat->flags & AVFMT_NOFILE)) { 
        int status = avio_open2(&formatContext->pb, formatContext->url, AVIO_FLAG_WRITE, nullptr, nullptr);
        if (status < 0) {
            return status;
        }
    }

    return 0;
}


int FfmpegStreamer::initCodecContext() {
    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);

    codecContext = avcodec_alloc_context3(codec);

    codecContext->width = 640;
    codecContext->height = 360;
    codecContext->pix_fmt = AV_PIX_FMT_YUV420P;
    codecContext->time_base = {1, 60}; 
    codecContext->max_b_frames = 0;   
    codecContext->gop_size = 30; 

    codecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    
    av_opt_set(codecContext->priv_data, "tune", "zerolatency", 0);
    av_opt_set(codecContext->priv_data, "preset", "ultrafast", 0);

    return avcodec_open2(codecContext, codec, nullptr);
}


int FfmpegStreamer::initSwsContext() {
    swsContext = sws_getContext(640, 360, AV_PIX_FMT_BGR24, 
                                640, 360, AV_PIX_FMT_YUV420P, 
                                SWS_FAST_BILINEAR, 
                                nullptr, nullptr, nullptr);

    yuvFrame->format = AV_PIX_FMT_YUV420P;
    yuvFrame->width  = 640;
    yuvFrame->height = 360;

    av_frame_get_buffer(yuvFrame, 0);
    return 0;
    }


int FfmpegStreamer::sendFrame(const cv::Mat& wrapper) {
    if (wrapper.empty() || wrapper.type() != CV_8UC3) {
        return -1;
    }
    const uint8_t* srcData[1] = { wrapper.data };
    int srcLinesize[1] = { static_cast<int>(wrapper.step[0]) };

    sws_scale(swsContext, 
              srcData, 
              srcLinesize, 
              0, 
              wrapper.rows,
              yuvFrame->data, 
              yuvFrame->linesize);

    yuvFrame->pts = frame_pts++;

    int ret = avcodec_send_frame(codecContext, yuvFrame);
    if (ret < 0) {
        return ret;
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(codecContext, encodedFrame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            return ret;
        }

        av_packet_rescale_ts(encodedFrame, codecContext->time_base, formatContext->streams[0]->time_base);
        encodedFrame->stream_index = 0;

        // Отправляем пакет в сокет
        av_interleaved_write_frame(formatContext, encodedFrame);
        av_packet_unref(encodedFrame);
    }

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