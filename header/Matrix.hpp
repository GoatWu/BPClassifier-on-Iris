//
//  Matrix.hpp
//  data_process
//
//  Created by GoatWu on 2020/5/12.
//  Copyright © 2020 吴朱冠宇. All rights reserved.
//

#ifndef Matrix_hpp
#define Matrix_hpp

#include <bits/stdc++.h>
using namespace std;

struct Matrix {
    int r, c;
    vector< vector<double> > v;
    Matrix() {}
    Matrix(int r, int c) : r(r), c(c) {
        
    }
};

#endif /* Matrix_hpp */
