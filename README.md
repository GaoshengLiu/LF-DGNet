# Disparity-Guided Light Field Image Super-Resolution via Feature Modulation and Recalibration

This repository contains official pytorch implementation of Disparity-Guided Light Field Image Super-Resolution via Feature Modulation and Recalibration, an early acceped paper in IEEE transactions on Broadcasting, 2023, by Gaosheng Liu, Huanjing Yue, Kun Li, and Jingyu Yang.
![Network](https://github.com/GaoshengLiu/LF-DGNet/blob/main/fig/network.jpg)  

## Dataset
We use the processed data by [LF-DFnet](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9286855), including EPFL, HCInew, HCIold, INRIA and STFgantry datasets for training and testing. Please download the dataset in the official repository of [LF-DFnet](https://github.com/YingqianWang/LF-DFnet).
## Code
### Dependencies
* Ubuntu 18.04
* Python 3.6
* Pyorch 1.3.1 + torchvision 0.4.2 + cuda 92
* Matlab
### Prepare Test Data
* To generate the test data, please first download the five datasets and run:
  ```matlab
  GenerateTestData.m
### Test
* Run:
  ```python
  python test.py
### Visual Results
* To merge the Y, Cb, Cr channels, run:
  ```matlab
  GenerateResultImages.m
## Citation
If you find this work helpful, please consider citing the following papers:<br> 
```Citation
@article{liu2023disparity,
  title={Disparity-Guided Light Field Image Super-Resolution via Feature Modulation and Recalibration},
  author={Liu, Gaosheng and Yue, Huanjing and Li, Kun and Yang, Jingyu},
  journal={IEEE Transactions on Broadcasting},
  year={2023},
  publisher={IEEE}
}
```
## Acknowledgement
Our work and implementations are inspired and based on the following projects: <br> 
[LF-DFnet](https://github.com/YingqianWang/LF-DFnet)<br> 
[LF-InterNet](https://github.com/YingqianWang/LF-InterNet)<br> 
We sincerely thank the authors for sharing their code and amazing research work!


