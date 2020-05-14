//
//  data_process.hpp
//  data_process
//
//  Created by GoatWu on 2020/5/12.
//  Copyright © 2020 吴朱冠宇. All rights reserved.
//

#ifndef data_process_hpp
#define data_process_hpp

#include "Matrix.hpp"
using namespace std;
const double eps = 1e-7;

class DataFrame {
public:
    Matrix X_train, Y_train, X_test, Y_test;
    int tot;
    vector<string> str;
    void read_file(const char *name);
    void init();
    
    map<string, int> ma;
    map<int, string> am;
    vector<string> tostring(Matrix &Y);
    vector<string> tostring(vector<int> &Y);
    string ans(int id);
    void show();
    
private:
    struct Data {
        vector<double> v;
        string label;
    };
    
    vector<Data> train, test;
    
    template <class T>
    void shuffle(vector<T> &a);
    bool isdouble(string s);
    void toOnehot(vector<Data> &d, Matrix &Y);
    void Normalize(vector<Data> &d, Matrix &X);
    void deal_data(vector<string> &vs, Matrix &X, Matrix &Y);
    
    
};

#endif /* data_process_hpp */
