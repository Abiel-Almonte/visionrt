#pragma once

#include <cstdint>
#include <fcntl.h>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <cuda_runtime.h>

#include "buffers.hpp"
#include "utils.hpp"
#include "v4l2.hpp"

struct CameraFMT {
    double fps;
    uint32_t pixelformat;
    uint32_t width;
    uint32_t height;
    std::string description;
};

inline std::ostream& operator<<(std::ostream& os, const CameraFMT& fmt) {
    os << "Camera(width=" << fmt.width << ", height=" << fmt.height
       << ", fps=" << fmt.fps << ", format=" << fmt.description << ")";
    return os;
}

class Camera {
private:
    int fd_ = -1;
    std::vector<CameraFMT> formats_;
    std::optional<size_t> best_fmt_idx_;
    std::optional<size_t> current_fmt_idx_;
    CameraRingBuffer ring_;
    bool streaming_ = false;

    static int open_device(const char* path) {
        int fd = open(path, O_RDWR);
        if (fd == -1) {
            std::ostringstream ss;
            ss << "Failed to open camera: " << strerror(errno);
            logging::error(ss);
            throw std::runtime_error("Failed to open camera device");
        }
        return fd;
    }

    void validate_capabilities() {
        auto caps = v4l2::query_capabilities(fd_);

        std::ostringstream ss;
        ss << "Using camera: " << caps.card << " | Bus: " << caps.bus_info;
        logging::info(ss);

        if (!caps.supports_streaming) {
            logging::error("Camera does NOT support streaming");
            throw std::runtime_error("Camera does not support streaming");
        }
        logging::info("Camera supports streaming");

        if (!caps.supports_ext_pix_format) {
            logging::error("Camera does NOT support pixformat");
            throw std::runtime_error("Camera does not support extended pixel formats");
        }
        logging::info("Camera supports pixformat");
    }

    void discover_formats() {
        double max_score = 0;

        for (const auto& v4l2_fmt : v4l2::enumerate_formats(fd_)) {
            for (const auto& interval : v4l2_fmt.intervals) {
                CameraFMT fmt{
                    interval.fps,
                    v4l2_fmt.pixelformat,
                    interval.width,
                    interval.height,
                    v4l2_fmt.description,
                };

                double score = fmt_score(fmt.fps, fmt.width, fmt.height);
                if (score > max_score) {
                    max_score = score;
                    best_fmt_idx_ = formats_.size();
                }

                formats_.push_back(std::move(fmt));
            }
        }
    }

public:
    explicit Camera(const char* device)
        : fd_(open_device(device))
        , ring_(fd_)
    {
        validate_capabilities();
        discover_formats();
        if (best_fmt_idx_) {
            set_format(*best_fmt_idx_);
        }
    }

    ~Camera() {
        try { stop_streaming(); } catch (...) {}
        if (fd_ >= 0) close(fd_);
    }

    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;


    void list_formats() const {
        std::ostringstream ss;
        ss << "Available formats:\n";
        for (size_t i = 0; i < formats_.size(); i++) {
            ss << "  [" << i << "] " << formats_[i];
            if (best_fmt_idx_ && i == *best_fmt_idx_) {
                ss << " (BEST)";
            }
            ss << "\n";
        }
        logging::info(ss);
    }

    void set_format(size_t index) {
        if (index >= formats_.size()) {
            throw std::out_of_range("Format index out of range");
        }

        const auto& fmt = formats_[index];
        v4l2::set_format(fd_, fmt.pixelformat, fmt.width, fmt.height);

        current_fmt_idx_ = index;
        std::ostringstream ss;
        ss << "Camera format set: " << fmt;
        logging::info(ss);
    }

    void print_format() const {
        if (!current_fmt_idx_) {
            throw std::runtime_error("No format set");
        }
        std::ostringstream ss;
        ss << "Current format: " << formats_[*current_fmt_idx_];
        logging::info(ss);
    }

    uint32_t width() const {
        if (!current_fmt_idx_) throw std::runtime_error("No format set");
        return formats_[*current_fmt_idx_].width;
    }

    uint32_t height() const {
        if (!current_fmt_idx_) throw std::runtime_error("No format set");
        return formats_[*current_fmt_idx_].height;
    }

    [[nodiscard]] bool is_streaming() const noexcept {
        return streaming_;
    }

    void start_streaming() {
        if (streaming_) return;
        ring_.start_streaming();
        streaming_ = true;
    }

    void stop_streaming() {
        if (!streaming_) return;
        ring_.stop_streaming();
        streaming_ = false;
    }

    void close_camera() {
        stop_streaming();
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    torch::Tensor capture_frame(int buffer_idx) const {
        void* src = ring_.buffer_start(buffer_idx);
        size_t size = ring_.buffer_length(buffer_idx);

        auto options = torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(torch::kCUDA);

        auto frame = torch::empty({
            static_cast<int64_t>(height()),
            static_cast<int64_t>(width()),
            2
        }, options);

        cudaMemcpy(frame.data_ptr(), src, size, cudaMemcpyHostToDevice);
        return frame;
    }

    Camera& __iter__() {
        return *this;
    }

    torch::Tensor __next__() {
        if (!is_streaming()) {
            start_streaming();
        }

        int idx = ring_.dequeue_buffer();
        if (idx == -1) {
            throw std::runtime_error("Failed to dequeue buffer");
        }

        auto frame = capture_frame(idx);
        ring_.queue_buffer(idx);
        return frame;
    }

    Camera& stream() {
        return *this;
    }

    std::string __repr__() const {
        if (!current_fmt_idx_) {
            return "Camera(no format set)";
        }
        std::ostringstream oss;
        oss << formats_[*current_fmt_idx_];
        return oss.str();
    }
};
