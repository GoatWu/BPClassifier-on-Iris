//
//  classifier.hpp
//  data_process
//
//  Created by GoatWu on 2020/5/13.
//  Copyright © 2020 吴朱冠宇. All rights reserved.
//

#ifndef classifier_hpp
#define classifier_hpp

#include "Matrix.hpp"
using namespace std;

class BPClassifier
{
private:
    const int LAYER = 3;     // 3 layers
    const int NUM = 10;      // the limit of neurons in each layer
    
    int iters;               // max training iteration
    double eta_w;            // learning rate for weight
    double eta_b;            // learning rate for bias
    
    int in_num;              // neurons number in input layer
    int hd_num;              // neurons number in hidden layer
    int ou_num;              // neurons number in output layer
    
    double t0,t1;
    
    double ***w;             // weight between neurons
    double **b;              // bias for neurons
    double **s;              // output for neurons
    double **delta;          // delta for neurons
    
    void get_num(const Matrix&, const Matrix&);              // get each layer's neurons number
    void generate_array(double***, int, int);                // generate 2-d array
    void generate_array(double****, int, int, int);          // generate 3-d array
    void random_start();                                     // give w/b random starting point
    void initialize_network(int);                            // network initialization
    void forward_propagation();                              // calculate s
    void calculate_delta(const vector<double>&);             // calculate delta
    void improve_network(int);                               // update w/b
    void backward_propagation(const vector<double>&, int);   // bp
    void record_network(string);                             // record num/w/b to txt
    void read_network(string);
    void free_array();                                       // read num/w/b from txt
    
    int forecast(const vector<double>&);                     // forecast the kind of input
    
    double random_01();                                      // return random float in [0,1]
    double sigmoid(double);                                  // digmoid function
    
    // calculate accuracy
    //double calculate_accuracy(const Matrix&, const Matrix&, int, int);
    double calculate_accuracy(const Matrix &X, const Matrix &Y, vector<int> v);
    
public:
    BPClassifier(int iters = 1000, double eta_w = 1e-1, double eta_b = 1e-1);
    ~BPClassifier();
    void fit(const Matrix&, const Matrix&);
    vector<int> predict(const Matrix&);
};

#endif /* classifier_hpp */
