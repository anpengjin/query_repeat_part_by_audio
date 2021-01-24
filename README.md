# 功能
传入多段音频，找出其中相同的部分，并给出各个音频相同部分的起止时间

# 环境
Python=3.8.3  
librosa=0.8.0  
numpy=1.19.2  
scipy=1.4.1  
pandas=1.0.5  
soundfile=0.10.3  
matplotlib=3.3.0  

# 音频特征
采样率：16000  
比特数：16bit  
声道：单声道  

# 原理
1. 倒排索引  
2. Shazam音频指纹提取算法  

# 运行

1 安装所需模块：pip install -r requirements.txt，然后将多个音频放入audio_data文件夹下  

2 首先从百度网盘里将数据下载完毕放入audio_data文件夹里(链接：https://pan.baidu.com/s/1YZWSFYUciFCeF6QyedMCOA 
提取码：anpj )  
然后直接运行python main.py可将结果打印在屏幕上  

或者运行./main.sh，结果会在log里  
log中会出现如下内容：  
path: ./audio_data/66.wav  sr: 16000  duration: 959.0  feature.shape: (34479, 3)  
path: ./audio_data/70.wav  sr: 16000  duration: 204.0  feature.shape: (6784, 3)  
path: ./audio_data/72.wav  sr: 16000  duration: 459.0  feature.shape: (17897, 3)  
target_advise_list of 66.wav: ['70', '72'] # 若音频数量过大，可通过倒排索引快速定位目标音频  
['66', 104.8875, 119.075] ['70', 2.7375, 16.925] 解释：音频66的第104秒到119秒和音频70的第2秒到第16秒是相同的内容（广告）  
['66', 895.975, 908.9625] ['72', 430.9375, 443.9375]  
['66', 90.4875, 104.2625] ['72', 415.45, 429.2375]  
['66', 940.5, 952.9375] ['72', 130.475, 142.9]  
target_advise_list of 70.wav: ['72', '66']  
['70', 2.9375, 16.925] ['72', 355.05, 369.0375]  
['70', 2.7375, 16.925] ['66', 104.8875, 119.075]  
target_advise_list of 72.wav: ['70', '66']  
['72', 355.05, 369.0375] ['70', 2.9375, 16.925]  
['72', 430.9375, 443.9375] ['66', 895.975, 908.9625]  
['72', 415.45, 429.2375] ['66', 90.4875, 104.2625]  
['72', 130.475, 142.9] ['66', 940.5, 952.9375]  
若想中途停止，运行./stop.sh  

# 参考
[1]. https://www.toptal.com/algorithms/shazam-it-music-processing-fingerprinting-and-recognition  
[2]. https://zhuanlan.zhihu.com/p/75360272  
[3]. https://github.com/lukemcraig/AudioSearch
[4]. [An Industrial-Strength Audio Search Algorithm](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf)