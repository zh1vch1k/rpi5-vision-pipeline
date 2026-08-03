#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"

namespace config {
    using StringDict = std::unordered_map<std::string, std::string>;
    using IntDict = std::unordered_map<std::string, int>;

    struct NetworkConfig {
        std::string ip;
        std::string port;
    };
        
    struct VideoConfig {
        int width;
        int height;
        int channels;
        int fps;
        };
        
    struct IPC {
        std::string bufferName;
        std::string semaphoreName;
    };

    struct Config {
        NetworkConfig network;
        VideoConfig video;
        IPC ipc;
    };


    inline Config getContext(const char* filename) {
        std::ifstream config(filename);
        if (!config.is_open()) {
            std::cerr << "[Config] Failed to open config file: " << filename << "\n"; 
            return Config{};
        }
        
        nlohmann::json ctx;
        config >> ctx;
        config.close();

        auto network = ctx["network"].get<StringDict>();
        auto video   = ctx["video"].get<IntDict>();
        auto ipc     = ctx["ipc"].get<StringDict>();

        NetworkConfig network_struct = NetworkConfig {network["ip"], network["port"]};
        VideoConfig video_struct = VideoConfig {video["width"], video["height"], video["channels"], video["fps"]};
        IPC ipc_struct = IPC {ipc["buffer_path"], ipc["semaphore_name"]};
        
        return Config {network_struct, video_struct, ipc_struct};
    }
}

