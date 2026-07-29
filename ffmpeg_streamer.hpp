extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

class FfmpegStreamer{
private: 



// SwsContext* swsCtx = sws_getContext(
//     640, 640, AV_PIX_FMT_BGR24,     
//     640, 640, AV_PIX_FMT_YUV420P, 
//     SWS_BILINEAR, nullptr, nullptr, nullptr
// );

// const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
// AVCodecContext* codecCtx = avcodec_alloc_context3(codec);

// AVFormatContext* formatCtx = nullptr;
// avformat_alloc_output_context2(&formatCtx, nullptr, "rtsp", "rtsp://localhost:8554/live");

public: 
};


