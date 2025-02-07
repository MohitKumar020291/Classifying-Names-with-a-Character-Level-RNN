#ifndef SET_DEVICE_HPP
#define SET_DEVICE_HPP

#include <iostream>

int is_cuda_available() {
    int output = std::system("nvcc --version");
    if (!output == 1) {
        std::cout << "CUDA is found" << "\n";
    } else {
        std::cerr << "CUDA is not found : running nvcc --version gives some error, error code is " << output << "\n";
    }
    return !output;
}

#endif