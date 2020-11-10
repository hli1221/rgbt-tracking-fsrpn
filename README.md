# FSRPN for VOT2019-RGBT
FSRPN (Fuison SiamRPN tracker), 4th in public dataset.  

Thanks to Pro. Xiao-Jun Wu*, Pro. Josef Kittler, [Dr. Tianyang Xu](https://github.com/XU-TIANYANG), Xuefeng Zhu, Yunkun Li

In FSRPN, spatial attention-based fusion strategy is applied to Siamese CNN framework. The deep features extracted by ResNet-50 from RGB and infrared images, are fused by this strategy to get more accurate and more plentiful information of object. Then, these fused deep features are utilized to track object by RPN-based network.


[VOT2019-RGBT subchallenge](https://www.votchallenge.net/vot2019/index.html)  
[The Seventh Visual Object Tracking VOT2019 Challenge Results](http://prints.vicos.si/publications/375)

## Requirements
Python  >= 3.6  
Pytorch >= 0.4.1  


# Tracker description:

The FSRPN exploits siamese CNN framework and spatial attention-based fusion strategy [1] for tracking. ResNet-50 [2] is used to extract multi-layers (three layers) deep features from RGB and infrared images, respectively. Before feeding into region proposal sub-networks, these features are fused by spatial attention-based fusion strategy based on l_1-norm in each layer. Then, these fused deep features and the region proposal sub-network are utilized to RPN networks to determine the position of object. The FSRPN is based on SiamRPN++ [3].

The fusion strategy comes [from DenseFuse](https://github.com/hli1221/imagefusion_densefuse).


[1] Li H, Wu X J. Densefuse: A fusion approach to infrared and visible images[J]. IEEE Transactions on Image Processing, 2018, 28(5): 2614-2623.  
[2] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.  
[3] Li B, Wu W, Wang Q, et al. SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks[J]. arXiv preprint arXiv:1812.11703, 2018.  



# The pre-trained model file

Please copy the model file('model.pth') to 'FSRPN/siamrpn_model/siamrpn_r50'.

The pre-trained model can be downloaded from [google drive](https://drive.google.com/file/d/1xoebTW6NLzdZGEYfu4jJb5BXxcY7xvx2/view?usp=sharing) and [baidu yun](https://pan.baidu.com/s/1gH_aEKuJT-yIrXOJe4CKeA) with extraction code '4bhx'.


# Run tracker

run_tracker_siamrpn.py


