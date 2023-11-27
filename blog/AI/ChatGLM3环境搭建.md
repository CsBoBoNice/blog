# ChatGLM3环境搭建

[toc]

## 官方仓库

### ChatGLM3 Github官方仓库 : 

```
https://github.com/THUDM/ChatGLM3
```

### 工程官方仓库地址

```
https://github.com/THUDM/ChatGLM3/tree/main
```

## 环境搭建

### Anaconda安装

#### Anaconda介绍

Anaconda是一个Python编程语言的开发环境，它包含了众多科学计算和数据分析库
Anaconda提供了一个方便的环境管理器

安装好Anaconda后会自带一个Anaconda Prompt用于执行命令行

使用conda命令很方便管理代码运行环境

#### Anaconda 镜像地址

清华大学开源软件镜像站:
```
https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/

```

Anaconda清华大学开源软件镜像站下载地址:
```
https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/
```

*选择下载Anaconda3-2023.09-0-Windows-x86_64.exe*

>注意安装时选上加入环境变量

若未选择则手动加入添加环境变量
添加环境变量:

```
C:\ProgramData\anaconda3
C:\ProgramData\anaconda3\Scripts\
C:\ProgramData\anaconda3\Library\bin
C:\ProgramData\anaconda3\Library\mingw-w64\bin
```

在命令行窗口输入conda测试



##### anaconda换源

参考网址
```
https://blog.csdn.net/weixin_49703503/article/details/128360909
```

打开Anaconda Prompt终端输入
在家目录生成名为 .condarc 的配置文件：
```
conda config --set show_channel_urls yes，
```

修改文件
```
C:\Users\csbobo\.condarc
```
改为

```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
``` 
  
打开终端运行

```
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

#### conda命令

##### 创建名为 env_name 的虚拟环境

创建名为 env_name 的虚拟环境：
```
conda create --name env_name
```

创建名为 env_name 的虚拟环境并同时安装 python3.7 ：
```
conda create --name env_name python=3.7
```

删除名为 env_name 的虚拟环境：
```
conda remove --name env_name --all
```

复制名为 env_name 的虚拟环境：
```
conda create --name env_name_old --clone env_name_new
```
PS：Anaconda没有重命名虚拟环境的操作，若要重命名虚拟环境，需要结合复制和删除虚拟环境两个命令实现。

##### 激活虚拟环境

激活名为 env_name 的虚拟环境：
```
conda activate env_name
```

##### 查看当前虚拟环境列表

```
conda env list 
```

或 
```
conda info -e
```

##### 给虚拟环境装包
指定虚拟环境名进行装包：
```
conda install -n env_name package_name
```

激活虚拟环境，并在该虚拟环境下装包：
```
conda activate env_name

conda install package_name
```

安装指定版本号的包：
```
conda install peckage_name==x.x
```

### 安装cuda环境和cuDNN

#### cuda和cuDNN介绍

##### 什么是CUDA
>CUDA(ComputeUnified Device Architecture)，是显卡厂商NVIDIA推出的运算平台。 CUDA是一种由NVIDIA推出的通用并行计算架构，该架构使GPU能够解决复杂的计算问题。

##### 什么是CUDNN
>NVIDIA cuDNN是用于深度神经网络的GPU加速库。它强调性能、易用性和低内存开销。NVIDIA cuDNN可以集成到更高级别的机器学习框架中，如谷歌的Tensorflow、加州大学伯克利分校的流行caffe软件。简单的插入式设计可以让开发人员专注于设计和实现神经网络模型，而不是简单调整性能，同时还可以在GPU上实现高性能现代并行计算。

##### CUDA与CUDNN的关系
>CUDA看作是一个工作台，上面配有很多工具，如锤子、螺丝刀等。cuDNN是基于CUDA的深度学习GPU加速库，有了它才能在GPU上完成深度学习的计算。它就相当于工作的工具，比如它就是个扳手。但是CUDA这个工作台买来的时候，并没有送扳手。想要在CUDA上运行深度神经网络，就要安装cuDNN，就像你想要拧个螺帽就要把扳手买回来。这样才能使GPU进行深度神经网络的工作，工作速度相较CPU快很多。

参考网址
```
https://blog.csdn.net/qq_45041871/article/details/127950087
https://blog.csdn.net/weixin_44159487/article/details/103364034
```

#### CUDA安装

安装前记得更新显卡驱动

cuda安装11.8版本

CUDA下载安装 ：
```
https://developer.nvidia.com/cuda-downloads
https://developer.nvidia.com/cuda-toolkit-archive
```
>安装时选择自定义(不要选择精简安装)

windows命令行 查看信息
```
nvidia-smi
nvcc -V
```

#### CUDNN安装

国内cuDNN下载地址
```
https://developer.nvidia.cn/rdp/cudnn-archive
```

国外下载地址
```
https://developer.nvidia.com/rdp/cudnn-download
```
>直接下载11.8对应的最新版本Download cuDNN v8.9.6 (November 1st, 2023), for CUDA 11.x

将文件夹cuDNN压缩包解压
将文件夹cuDNN内文件拷贝到CUDA对应文件夹
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\
```

添加环境变量path
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib
```
进到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\demo_suite目录

在文件夹路径输入cmd
```
.\deviceQuery.exe
.\bandwidthTest.exe
```
确认是否正常


### 官方demo环境搭建

#### 基本环境搭建

##### 创建conda环境

先通过Anaconda Prompt程序进入
输入命令进入目录
```
D:
cd ChatGLM3_6B\ChatGLM3-main
```

先通过Anaconda Prompt程序进入
输入命令进入目录
```
D:
cd ChatGLM3_6B\ChatGLM3-main\composite_demo
```

>进入环境
```
conda activate chatglm3-demo
```

>安装依赖(可能失败多次重复)
```
pip3 install -r requirements.txt
```
(注意这里要用pip3)

##### 在cuad环境安装pyctorch
参考网址
```
https://blog.csdn.net/m0_46948660/article/details/129205116
```

查看pyctorch对应版本地址
```
https://pytorch.org/get-started/previous-versions/
```

得到下载地址
```
https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/
```
找到对应版本pytorch-2.1.1-py3.10_cuda11.8_cudnn8_0.tar.bz2

>进入环境（注意安装前要先进入环境）
```
conda activate chatglm3-demo
```

使用命令安装pyctorch:
```
conda install https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/pytorch-2.1.1-py3.10_cuda11.8_cudnn8_0.tar.bz2
```

自动安装依赖包
```
conda install pytorch
```

测试环境是否正常
>进入环境
```
conda activate chatglm3-demo
```

命令行输入python
```
import torch
print("是否可用:",torch.cuda.is_available())
print("GPU数量:",torch.cuda.device_count())
print("torch查看CUDA版本:",torch.version.cuda)
print("GPU索引号:",torch.cuda.current_device())
print("GPU名称:",torch.cuda.get_device_name(0))
exit()
```



#### composite_demo环境搭建(带web界面)

根据ChatGLM3-main\composite_demo下的README.md设置

先通过Anaconda Prompt程序进入
输入命令进入目录
```
D:
cd ChatGLM3_6B\ChatGLM3-main\composite_demo
```

执行以下命令新建一个 conda 环境并安装所需依赖：
>创建环境
```
conda create -n chatglm3-demo python=3.10
```
>进入环境
```
conda activate chatglm3-demo
```

>安装依赖(可能失败多次重复)
```
pip3 install -r requirements.txt
```
(注意这里要用pip3)

>使用 Code Interpreter 还需要安装 Jupyter 内核：
```
ipython kernel install --name chatglm3-demo --user
```
返回的安装路径
```
Installed kernelspec chatglm3-demo in C:\Users\csbobo\AppData\Roaming\jupyter\kernels\chatglm3-demo
```

将模型目录添加到环境变量
```
MODEL_PATH
D:\ChatGLM3_6B\models\THUDM_chatglm3-6b
```

>在本地加载模型并启动 demo：
```
streamlit run main.py
```

>查看环境安装的版本
```
conda list
```

#### basic_demo环境搭建(命令行界面)

安装整个官方工程所需环境(上面是单个demo的环境)
先通过Anaconda Prompt程序进入
输入命令进入目录
```
D:
cd ChatGLM3_6B\ChatGLM3-main
```

先通过Anaconda Prompt程序进入
输入命令进入目录
```
D:
cd ChatGLM3_6B\ChatGLM3-main\composite_demo
```

>进入环境
```
conda activate chatglm3-demo
```

>安装依赖(可能失败多次重复)
```
pip3 install -r requirements.txt
```
(注意这里要用pip3)

>进入demo目录测试
```
cd ChatGLM3_6B\ChatGLM3-main\basic_demo
```

修改代码
增加

```
MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
PT_PATH = os.environ.get('PT_PATH', None)
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
```

修改
```
#tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
```
修改量化等级
```
#model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).quantize(4).cuda()
```

>运行代码

```
python cli_demo.py
```

## chatglm.cpp环境部署

### chatglm.cpp介绍

亮点：
* 基于ggml的纯C++实现，工作方式与llama.cpp相同。
* 加速内存高效的CPU推理，带有int4/int8量化，优化KV缓存和并行计算。
* 带有多元writer效果的流式生成。
* 支持Python绑定，Web演示，API服务器等功能。 支持矩阵：
* 硬件：x86/arm CPU，NVIDIA GPU，Apple Silicon GPU
* 平台：Linux，MacOS，Windows
* 模型：ChatGLM-6B，ChatGLM2-6B，ChatGLM3-6B，CodeGeeX2，Baichuan-13B，Baichuan-7B，Baichuan-13B，Baichuan2，InternLM

### 官方项目地址
>官方项目地址
```
https://github.com/THUDM/ChatGLM3
```

>C++的chatGLM项目地址
```
https://github.com/li-plus/chatglm.cpp
```

>下载官方模型
```
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
```

>国内镜像模型网站
```
https://aliendao.cn/#/
```

### 准备工作

将ChatGLM.cpp仓库克隆到本地计算机：
```sh
git clone --recursive https://github.com/li-plus/chatglm.cpp.git && cd chatglm.cpp
```

如果您在克隆仓库时忘记添加--recursive标志，请在ChatGLM.cpp文件夹中运行以下命令：
```sh
git submodule update --init --recursive
```

>创建环境
```
conda create -n chatglm_cpp python=3.11
```

>进入环境
```
conda activate chatglm_cpp
```

安装环境
```
pip3 install torch tabulate tqdm transformers accelerate sentencepiece
```

安装cmake
```
conda install cmake
```

### 量化模型

使用convert.py将ChatGLM-6B转换为量化GGML格式。例如，将fp16原模型转换为q4_0（量化int4）GGML模型，运行：
```sh
python3 chatglm_cpp/convert.py -i THUDM/chatglm-6b -t q4_0 -o chatglm-ggml.bin
```

原始模型 (`-i <model_name_or_path>`)可以是Hugging Face模型名称或您预先下载的模型的本地路径。目前支持的模型有：
* ChatGLM-6B: `THUDM/chatglm-6b`, `THUDM/chatglm-6b-int8`, `THUDM/chatglm-6b-int4`
* ChatGLM2-6B: `THUDM/chatglm2-6b`, `THUDM/chatglm2-6b-int4`
* ChatGLM3-6B: `THUDM/chatglm3-6b`
* CodeGeeX2: `THUDM/codegeex2-6b`, `THUDM/codegeex2-6b-int4`
* Baichuan & Baichuan2: `baichuan-inc/Baichuan-13B-Chat`, `baichuan-inc/Baichuan2-7B-Chat`, `baichuan-inc/Baichuan2-13B-Chat`

您可以自由尝试以下任何量化类型，通过指定 `-t <type>`：
* `q4_0`: 4位整数量化，带有fp16缩放因子.
* `q4_1`: 4位整数量化，带有fp16缩放因子和最小值.
* `q5_0`: 5位整数量化，带有fp16缩放因子
* `q5_1`: 5位整数量化，带有fp16缩放因子和最小值.
* `q8_0`: 8位整数量化，带有fp16缩放因子.
* `f16`: 不进行量化的半精度浮点权重
* `f32`: 不进行量化的单精度浮点权重。

对于LoRa模型，添加`-l <lora_model_name_or_path>`标志以将您的LoRa权重合并到基础模型中。

#### chatglm3-demo模型转换

>4 位整数量化，带有 fp16 缩放
```
python .\chatglm_cpp\convert.py -i D:\ChatGLM3_6B\models\chatglm3-6b -t q4_0 -o D:\ChatGLM3_6B\models\chatglm-ggml\chatglm-ggml_q4_0.bin
```

>8 位整数量化，带有 fp16 缩放。
```
python .\chatglm_cpp\convert.py -i D:\ChatGLM3_6B\models\chatglm3-6b -t q8_0 -o D:\ChatGLM3_6B\models\chatglm-ggml\chatglm-ggml_q8_0.bin
``` 

>f16：没有量化的半精度浮点权重
```
python .\chatglm_cpp\convert.py -i D:\ChatGLM3_6B\models\chatglm3-6b -t f16 -o D:\ChatGLM3_6B\models\chatglm-ggml\chatglm-ggml_f16.bin
```

>f32：没有量化的单精度浮点权重
```
python .\chatglm_cpp\convert.py -i D:\ChatGLM3_6B\models\chatglm3-6b -t f32 -o D:\ChatGLM3_6B\models\chatglm-ggml\chatglm-ggml_f32.bin
```

#### chatglm3-6b-base模型转换

>4 位整数量化，带有 fp16 缩放
```
python .\chatglm_cpp\convert.py -i D:\ChatGLM3_6B\models\chatglm3-6b-base -t q4_0 -o D:\ChatGLM3_6B\models\chatglm-ggml\chatglm3-6b-base-ggml_q4_0.bin
```

>8 位整数量化，带有 fp16 缩放。
```
python .\chatglm_cpp\convert.py -i D:\ChatGLM3_6B\models\chatglm3-6b-base -t q8_0 -o D:\ChatGLM3_6B\models\chatglm-ggml\chatglm3-6b-base-ggml_q8_0.bin
```
 
>f16：没有量化的半精度浮点权重
```
python .\chatglm_cpp\convert.py -i D:\ChatGLM3_6B\models\chatglm3-6b-base -t f16 -o D:\ChatGLM3_6B\models\chatglm-ggml\chatglm3-6b-base-ggml_f16.bin
```

>f32：没有量化的单精度浮点权重
```
python .\chatglm_cpp\convert.py -i D:\ChatGLM3_6B\models\chatglm3-6b-base -t f32 -o D:\ChatGLM3_6B\models\chatglm-ggml\chatglm3-6b-base-ggml_f32.bin
```

#### chatglm3-6b-32k模型转换

>转4 位整数量化，带有 fp16 缩放
```
python .\chatglm_cpp\convert.py -i D:\ChatGLM3_6B\models\chatglm3-6b-32k -t q4_0 -o D:\ChatGLM3_6B\models\chatglm-ggml\chatglm3-6b-32k-ggml_q4_0.bin
```

>8 位整数量化，带有 fp16 缩放。
```
python .\chatglm_cpp\convert.py -i D:\ChatGLM3_6B\models\chatglm3-6b-32k -t q8_0 -o D:\ChatGLM3_6B\models\chatglm-ggml\chatglm3-6b-32k-ggml_q8_0.bin
```
 
>f16：没有量化的半精度浮点权重
```
python .\chatglm_cpp\convert.py -i D:\ChatGLM3_6B\models\chatglm3-6b-32k -t f16 -o D:\ChatGLM3_6B\models\chatglm-ggml\chatglm3-6b-32k-ggml_f16.bin
```

>f32：没有量化的单精度浮点权重
```
python .\chatglm_cpp\convert.py -i D:\ChatGLM3_6B\models\chatglm3-6b-32k -t f32 -o D:\ChatGLM3_6B\models\chatglm-ggml\chatglm3-6b-32k-ggml_f32.bin
```

### 编译工程

>编译工程
```
cmake -B build
cmake --build build -j --config Release
```

>运行测试
```
.\build\bin\Release\main.exe -m D:\ChatGLM3_6B\models\chatglm-ggml\chatglm3-6b-32k-ggml_f16.bin -i
```

#### 编译在CPU上跑的bin文件
编译在CPU上跑的bin文件(效果很差)
(OpenBLAS 在 CPU 上提供加速。要启用它，请添加 CMake 标志 -DGGML_OPENBLAS=ON)

```
cmake -B build -DGGML_OPENBLAS=ON && cmake --build build -j
```

>运行测试
```
.\build\bin\Debug\main.exe -m D:\ChatGLM3_6B\models\chatglm-ggml\chatglm3-6b-32k-ggml_f16.bin -i
```

#### 编译在GPU上跑的bin文件
编译在GPU上跑的bin文件(效果非常好)
(cuBLAS 使用 NVIDIA GPU 加速 BLAS)
```
cmake -B build -DGGML_CUBLAS=ON && cmake --build build -j
```

运行测试
```
.\build\bin\Debug\main.exe -m D:\ChatGLM3_6B\models\chatglm-ggml\chatglm3-6b-32k-ggml_f16.bin -i
```

### chatglm.cpp使用Python 绑定

#### 介绍Python 绑定
>Python 绑定提供了类似于原始 Hugging Face ChatGLM(2)-6B 的面向高级的 chat 和 stream_chat 接口

#### python绑定 在CPU运行

从源代码安装Python绑定
在chatglm.cpp工程根路径下

>编译并安装chatglm_cpp包用于python
```
pip install .
```

>如果想卸载可以执行
```
pip uninstall ChatGLM-cpp
```

##### 测试运行 直接python运行
切换到chatglm.cpp\examples工程根路径下
测试运行
python
```
import chatglm_cpp
pipeline = chatglm_cpp.Pipeline("D:\ChatGLM3_6B\models\chatglm-ggml\chatglm-ggml_f16.bin")
pipeline.chat(["你好"])
```

##### 测试运行 运行cli_chat.py
在chatglm.cpp\examples工程根路径下
```
python cli_chat.py -m D:\ChatGLM3_6B\models\chatglm-ggml\chatglm-ggml_f16.bin -i
```

#### python绑定 在GPU运行

>修复找不到DLL的问题
在chatglm.cpp\chatglm_cpp目录下`__init__.py`文件加入以下代码
Python3.8之后需要通过os.add_dll_directory手动添加DLL搜索路径。
这里我们需要这两个CUDA的DLL：
你只需要在`__init__.py`文件的import chatglm_cpp._C as _C之前执行os.add_dll_directory(os.environ['CUDA_PATH'] + '/bin')即可。
>参考代码如下

```
import sys
if sys.version_info >= (3, 8) and sys.platform == "win32":
    import os
    if os.environ.get('CUDA_PATH') is not None:
        os.add_dll_directory(os.environ['CUDA_PATH'] + '/bin')

import chatglm_cpp._C as _C
```

>编译并安装chatglm_cpp包用于python
```
set "CMAKE_ARGS=-DGGML_CUBLAS=ON" && pip install . --force-reinstall -v
```

>如果想卸载可以执行
```
pip uninstall ChatGLM-cpp
```

##### 测试运行cli_chat
在chatglm.cpp\examples路径下
```
python cli_chat.py -m D:\ChatGLM3_6B\models\chatglm-ggml\chatglm3-6b-32k-ggml_q8_0.bin -i
python cli_chat.py -m D:\ChatGLM3_6B\models\chatglm-ggml\chatglm3-6b-32k-ggml_f16.bin -i
python cli_chat.py -m D:\ChatGLM3_6B\models\chatglm-ggml\chatglm3-6b-32k-ggml_f32.bin -i
```

##### 测试运行web_demo

>进入环境
```
conda activate chatglm_cpp
```

>安装gradio
```
pip install gradio
```

在chatglm.cpp\examples路径下
```
python web_demo.py -m D:\ChatGLM3_6B\models\chatglm-ggml\chatglm3-6b-32k-ggml_f16.bin
python web_demo.py -m D:\ChatGLM3_6B\models\chatglm-ggml\chatglm3-6b-32k-ggml_q8_0.bin
```

##### 测试运行langchain_client

在chatglm.cpp\examples路径下
```
python langchain_client.py -m D:\ChatGLM3_6B\models\chatglm-ggml\chatglm-ggml_f16.bin
```

安装依赖(GPU加速)
```
set "CMAKE_ARGS=-DGGML_CUBLAS=ON" && pip install chatglm-cpp[api]
```

启动 LangChain 的 API 服务器
```
set "MODEL=D:\ChatGLM3_6B\models\chatglm-ggml\chatglm-ggml_f16.bin" && uvicorn chatglm_cpp.langchain_api:app --host 127.0.0.1 --port 8000
```

安装langchain
```
pip install langchain
```

另外开命令行窗口运行API的客户端用于对话
```
python langchain_client.py
```

##### 相关但没用到的命令

>设置conda环境变量
```
conda env config vars set CMAKE_ARGS="-DGGML_CUBLAS=ON"
```

>移除环境变量
```
conda env config vars unset CMAKE_ARGS -n chatglm3-demo
```

>重新激活环境
```
conda activate chatglm3-demo
```

