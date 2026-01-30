#pragma once

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <linux/videodev2.h>
#include <sstream>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <vector>

class CameraRingBuffer {
private:
    struct Buffer {
        void* data = nullptr;
        size_t length = 0;
        v4l2_buffer v4l2_buf{};
    };

    int fd_;
    v4l2_buf_type type_;
    v4l2_memory memory_;
    std::vector<Buffer> buffers_;
    v4l2_buffer dequeue_buf_{};
    bool initialized_ = false;
    bool streaming_ = false;

    void request_buffers(size_t count) {
        v4l2_requestbuffers req{};
        req.type = type_;
        req.memory = memory_;
        req.count = count;

        if (ioctl(fd_, VIDIOC_REQBUFS, &req) == -1) {
            std::ostringstream ss;
            ss << "VIDIOC_REQBUFS failed: " << strerror(errno);
            throw std::runtime_error(ss.str());
        }

        buffers_.resize(req.count);
    }

    void map_buffers() {
        for (size_t i = 0; i < buffers_.size(); i++) {
            auto& buf = buffers_[i];

            buf.v4l2_buf.type = type_;
            buf.v4l2_buf.memory = memory_;
            buf.v4l2_buf.index = i;

            if (ioctl(fd_, VIDIOC_QUERYBUF, &buf.v4l2_buf) == -1) {
                unmap_buffers(i);
                std::ostringstream ss;
                ss << "VIDIOC_QUERYBUF failed for buffer " << i << ": " << strerror(errno);
                throw std::runtime_error(ss.str());
            }

            buf.length = buf.v4l2_buf.length;
            buf.data = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.v4l2_buf.m.offset);

            if (buf.data == MAP_FAILED) {
                buf.data = nullptr;
                unmap_buffers(i);
                std::ostringstream ss;
                ss << "mmap failed for buffer " << i << ": " << strerror(errno);
                throw std::runtime_error(ss.str());
            }
        }
    }

    void unmap_buffers(size_t count) {
        for (size_t i = 0; i < count && i < buffers_.size(); i++) {
            if (buffers_[i].data && buffers_[i].data != MAP_FAILED) {
                munmap(buffers_[i].data, buffers_[i].length);
                buffers_[i].data = nullptr;
            }
        }
    }

    void queue_all() {
        for (size_t i = 0; i < buffers_.size(); i++) {
            queue_buffer(i);
        }
    }

    void initialize() {
        if (initialized_) return;

        request_buffers(buffers_.size());
        map_buffers();

        dequeue_buf_.type = type_;
        dequeue_buf_.memory = memory_;

        initialized_ = true;
    }

public:
    explicit CameraRingBuffer(
        int fd,
        size_t num_buffers = 3,
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
        v4l2_memory memory = V4L2_MEMORY_MMAP
    )
        : fd_(fd)
        , type_(type)
        , memory_(memory)
        , buffers_(num_buffers)
    {}

    ~CameraRingBuffer() {
        try { stop_streaming(); } catch (...) {}
        unmap_buffers(buffers_.size());
    }

    CameraRingBuffer(const CameraRingBuffer&) = delete;
    CameraRingBuffer& operator=(const CameraRingBuffer&) = delete;
    CameraRingBuffer(CameraRingBuffer&&) = delete;
    CameraRingBuffer& operator=(CameraRingBuffer&&) = delete;

    void queue_buffer(size_t index) {
        if (index >= buffers_.size()) {
            throw std::out_of_range("Buffer index out of range");
        }

        if (ioctl(fd_, VIDIOC_QBUF, &buffers_[index].v4l2_buf) == -1) {
            std::ostringstream ss;
            ss << "VIDIOC_QBUF failed for buffer " << index << ": " << strerror(errno);
            throw std::runtime_error(ss.str());
        }
    }

    [[nodiscard]] int dequeue_buffer() {
        dequeue_buf_.type = type_;
        dequeue_buf_.memory = memory_;

        if (ioctl(fd_, VIDIOC_DQBUF, &dequeue_buf_) == -1) {
            return -1;
        }

        return static_cast<int>(dequeue_buf_.index);
    }

    void start_streaming() {
        if (streaming_) return;

        initialize();
        queue_all();

        if (ioctl(fd_, VIDIOC_STREAMON, &type_) == -1) {
            std::ostringstream ss;
            ss << "VIDIOC_STREAMON failed: " << strerror(errno);
            throw std::runtime_error(ss.str());
        }

        streaming_ = true;
    }

    void stop_streaming() {
        if (!streaming_) return;

        if (ioctl(fd_, VIDIOC_STREAMOFF, &type_) == -1) {
            std::ostringstream ss;
            ss << "VIDIOC_STREAMOFF failed: " << strerror(errno);
            throw std::runtime_error(ss.str());
        }

        streaming_ = false;
    }

    [[nodiscard]] bool is_streaming() const noexcept {
        return streaming_;
    }

    [[nodiscard]] size_t size() const noexcept {
        return buffers_.size();
    }

    [[nodiscard]] void* buffer_start(size_t index) const {
        if (index >= buffers_.size()) {
            throw std::out_of_range("Buffer index out of range");
        }
        return buffers_[index].data;
    }

    [[nodiscard]] size_t buffer_length(size_t index) const {
        if (index >= buffers_.size()) {
            throw std::out_of_range("Buffer index out of range");
        }
        return buffers_[index].length;
    }
};
