#include "set_device.hpp"
#include "Network.hpp"
#include "helper/print.hpp"

int main() {
    std::string device="cpu";
    int cuda = is_cuda_available();
    if (cuda) {
        device = "cuda";
    }
    std::cout << "running on " << device << "\n";

    int in_features = 2;
    int out_features = 3;
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(10, in_features);
    Linear linear(cuda, in_features, out_features, input, false);
    linear.backward();
    print(linear.weight_grads.rows());
    
    return 0;
}