

# Overview

基于OptiX官方提供的教程 （https://github.com/NVIDIA/OptiX_Apps）， 实现物体渲染， 用于前背景融合方法。


## 运行说明

(1)  需要先下载OptiX的SDK到本地路径， 下载链接：  https://developer.nvidia.com/designworks/optix/downloads/legacy, 下载版本：  NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64
    下载完成后，修改代码路径下/data/codes/optix_all/optix_my/OptiX_Apps/3rdparty/CMake/FindOptiX77.cmake中的 OPTIX77_PATH 为本地的SDK的实际路径！！
(2)  编译第三方库 fastgltf
git submodule update --init --recursive
cd  3rdparty/fastgltf/
mkdir build
mkdir install
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make -j8
make install


### 原版本的examples

cd  OptiX_Apps
mkdir build
cd build
CUDACXX=/usr/local/cuda-12.0/bin/nvcc  cmake -DCMAKE_BUILD_TYPE=Release  -DCMAKE_PREFIX_PATH=/data/codes/optix_all/optix_my/OptiX_Apps/3rdparty/fastgltf/install  -DCMAKE_CUDA_ARCHITECTURES=61  ..
make -j8

在build/bin路径下会生成可以测试的examples

### 新增加的物体渲染example

#### 编译
cd  OptiX_Apps/apps/render_my
mkdir  build
cd build
CUDACXX=/usr/local/cuda-12.0/bin/nvcc  cmake -DCMAKE_BUILD_TYPE=Release  -DCMAKE_PREFIX_PATH=/data/codes/optix_all/optix_my/OptiX_Apps/3rdparty/fastgltf/install  -DCMAKE_CUDA_ARCHITECTURES=61  ..
make  -j8
make install

在build/bin路径下会生成物体渲染的可执行程序render_my

#### 运行 
cd OptiX_Apps/apps/render_my/build
保存数据模式      ./bin/render_my  --compare 
可视化模式        ./bin/render_my











