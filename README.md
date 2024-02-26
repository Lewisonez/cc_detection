# SCAC: A Semi-Supervised Learning Approach for Cervical Abnormal Cell Detection 

To facilitate research in semi-supervised learning and cervical abnormal cell detection，we create an largely unlabeled cervical cytology dataset：  

**link：https://pan.baidu.com/s/1wWJY_TdyhtEI1cwQLBAPmA** 

**password：cmk2**

Our dataset includes 2471 WSI（Whole Slide Images）. These WSI were obtained using two different slide scanning devices, with 1204 from scanner 1 and 1267 from scanner 2.
By processing these WSI，hundreds of thousands of images can be obtained. 


Our main experimental results and weights are as follows：

|   Backbone | AP | AP@50 | AP@75 | weights |
| ------- | ------- | ------- | ------- | ------- |
| Resnet-50 | 26.9 |  48.7 |  27.0  |[link](https://pan.baidu.com/s/1PKfM1atP_n_KuLkPam3Slw?pwd=1111)   |
| Resnet-101 | 28.1 |  50.4 |  28.2  | [link](https://pan.baidu.com/s/1otrr1pqPeXWXxE1wEJphlQ?pwd=1111)  |
| PVT  | 28.8 |  50.8  |  28.9 |[link](https://pan.baidu.com/s/1rK4i7BOXsL8GAAl6SMK0vA?pwd=1111)  |
| Swin-Transformer | 29.4 |  51.5  |  29.5  | [link](https://pan.baidu.com/s/1VUV4mpZasWv0SKR5XJheAw?pwd=1111)   |

### Note
Our approach uses mmdetection, some modules and code refer to mmdetection(https://github.com/open-mmlab/mmdetection)
