# Alibaba-Cloud-German-AI-Challenge-2018

这是天池大数据的比赛<br/>
https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100067.5678.1.3e7731f5WP7NmY&raceId=231683<br/>

网络采用了L_Resnet_E_IR, 损失函数基于softmax entropy

数据集特点：<br>
类别分布不均衡<br>
某些通道的像素值远大于其他通道<br>

目前的参考方案：<br/>
1.采用端到端的网络，输出直接是label<br/>
2.先利用神经网络学习特征，然后获取神经网络最后一层的向量，利用传统分类器，如GBDT,LightGBM来分类
3.神经网络集成

