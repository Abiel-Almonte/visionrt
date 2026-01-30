#include <torch/torch.h>
#include <vector>

torch::Tensor launch_add_relu(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs
);

torch::Tensor launch_yuyv2rgb_chw(
    const torch::Tensor& yuyv,
    int height,
    int width,
    const std::vector<float>& scale,
    const std::vector<float>& offset
);
