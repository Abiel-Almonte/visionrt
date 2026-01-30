#pragma once

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <linux/videodev2.h>
#include <sstream>
#include <stdexcept>
#include <sys/ioctl.h>
#include <vector>

#include "utils.hpp"

namespace v4l2 {

struct Capabilities {
    std::string card;
    std::string bus_info;
    bool supports_streaming;
    bool supports_ext_pix_format;
};

struct FrameInterval {
    uint32_t width;
    uint32_t height;
    double fps;
};

struct Format {
    uint32_t pixelformat;
    std::string description;
    std::vector<FrameInterval> intervals;
};

inline Capabilities query_capabilities(int fd) {
    v4l2_capability caps{};
    if (ioctl(fd, VIDIOC_QUERYCAP, &caps) == -1) {
        std::ostringstream ss;
        ss << "VIDIOC_QUERYCAP failed: " << strerror(errno);
        throw std::runtime_error(ss.str());
    }

    return Capabilities{
        .card = std::string(reinterpret_cast<char*>(caps.card)),
        .bus_info = std::string(reinterpret_cast<char*>(caps.bus_info)),
        .supports_streaming = check_for_flag(caps.device_caps, V4L2_CAP_STREAMING),
        .supports_ext_pix_format = check_for_flag(caps.device_caps, V4L2_CAP_EXT_PIX_FORMAT),
    };
}

inline std::vector<FrameInterval> enumerate_frame_intervals(int fd, uint32_t pixelformat, uint32_t width, uint32_t height) {
    std::vector<FrameInterval> intervals;

    for (int idx = 0;; idx++) {
        v4l2_frmivalenum ival{};
        ival.pixel_format = pixelformat;
        ival.index = idx;
        ival.width = width;
        ival.height = height;

        if (ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &ival) == -1) {
            if (errno == EINVAL) break;
            std::ostringstream ss;
            ss << "VIDIOC_ENUM_FRAMEINTERVALS failed: " << strerror(errno);
            logging::error(ss);
            break;
        }

        if (ival.type != V4L2_FRMIVAL_TYPE_DISCRETE) continue;

        double fps = static_cast<double>(ival.discrete.denominator) / ival.discrete.numerator;
        intervals.push_back({width, height, fps});
    }

    return intervals;
}

inline std::vector<FrameInterval> enumerate_frame_sizes(int fd, uint32_t pixelformat) {
    std::vector<FrameInterval> all_intervals;

    for (int idx = 0;; idx++) {
        v4l2_frmsizeenum res{};
        res.pixel_format = pixelformat;
        res.index = idx;

        if (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &res) == -1) {
            if (errno == EINVAL) break;
            std::ostringstream ss;
            ss << "VIDIOC_ENUM_FRAMESIZES failed: " << strerror(errno);
            logging::error(ss);
            break;
        }

        if (res.type != V4L2_FRMSIZE_TYPE_DISCRETE) continue;

        auto intervals = enumerate_frame_intervals(fd, pixelformat, res.discrete.width, res.discrete.height);
        all_intervals.insert(all_intervals.end(), intervals.begin(), intervals.end());
    }

    return all_intervals;
}

inline std::vector<Format> enumerate_formats(int fd) {
    std::vector<Format> formats;

    for (int idx = 0;; idx++) {
        v4l2_fmtdesc desc{};
        desc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        desc.index = idx;

        if (ioctl(fd, VIDIOC_ENUM_FMT, &desc) == -1) {
            if (errno == EINVAL) break;
            std::ostringstream ss;
            ss << "VIDIOC_ENUM_FMT failed: " << strerror(errno);
            logging::error(ss);
            break;
        }

        if (!fmt_is_uncompressed(desc)) continue;

        Format fmt{
            .pixelformat = desc.pixelformat,
            .description = std::string(reinterpret_cast<char*>(desc.description)),
            .intervals = enumerate_frame_sizes(fd, desc.pixelformat),
        };

        if (!fmt.intervals.empty()) {
            formats.push_back(std::move(fmt));
        }
    }

    return formats;
}

inline void set_format(int fd, uint32_t pixelformat, uint32_t width, uint32_t height) {
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = pixelformat;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        std::ostringstream ss;
        ss << "VIDIOC_S_FMT failed: " << strerror(errno);
        throw std::runtime_error(ss.str());
    }
}

} // namespace v4l2
