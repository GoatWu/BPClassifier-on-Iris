# BPClassifier-on-Iris

XMU 2020 spring 面向对象程序设计（C++）BP neural network classifier practiced with iris.

误差反向传播学习算法（BP）实现Iris数据分类

class:
+ Matrix 实现矩阵类，作为数据载体。
+ DataFrame 实现数据的各种处理。
+ BPClassifier 基于神经网络的分类器，支持`fit`和`predict`。

结构：
```
|-- main.cpp
|-- classifier.cpp
|-- data_process.cpp 
|-- data   
|   |-- iris.data
|-- header
|   |-- classifier.hpp
|   |-- data_process.hpp
|   |-- Matrix.hpp
|-- model
```
