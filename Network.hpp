#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <eigen3/Eigen/Dense>
#include "helper/print.hpp"

class Linear {
    public:
    int in_features;
    int out_features;
    bool bias;
    Eigen::MatrixXf weights;
    Eigen::MatrixXf biases;
    Eigen::MatrixXf output;

    Eigen::MatrixXf input_data;

    //these grads are with respect to the output not loss
    Eigen::MatrixXf weight_grads;
    Eigen::MatrixXf input_grads;

    Linear(bool cuda, int in_features_, int out_features_, Eigen::MatrixXf input, bool bias) 
    : in_features(in_features_), out_features(out_features_), input_data(input) {
        this->weights = Eigen::MatrixXf::Random(in_features, out_features); //each column represents a neuron
        this->output = input * this->weights; //each column represents a data point
        if (bias) {
            this->biases = Eigen::MatrixXf::Random(out_features, 1).replicate(1, input.cols());
            this->output += this->biases;
        }
    }

    void backward() {
        // the multiplication chain changes
        // https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
        this->weight_grads = this->input_data.transpose();
        this->input_grads = this->weights.transpose();
    }
};

#endif