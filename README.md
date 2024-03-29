<div align="center">

# Rethinking Multi-Scale Representations in Deep Deraining Transformer

</div>

<!-- > Rethinking Multi-Scale Representations in Deep Deraining Transformer -->


## 🛠️ Training and Testing
1. Please put datasets in the folder `Datasets/`.
2. Follow the instructions below to begin training our model.
```
bash train.sh
```
Run the script then you can find the generated experimental logs in the folder `checkpoints`.

3. Follow the instructions below to begin testing our model.
```
python test.py
```
Run the script then you can find the output visual results in the folder `results/`.


## 🤖 Pre-trained Models
| Models | MSDT |
|:-----: |:-----: |
| Rain200L | [Google Drive](https://drive.google.com/file/d/1qk8pUq7oM4Z4v2X-qmWJpE2LmUuweL4_/view?usp=drive_link) / [Baidu Netdisk](https://pan.baidu.com/s/1jikJhCuv51bvkl9vF2AkKw?pwd=8ajd) (8ajd) 
| Rain200H | [Google Drive](https://drive.google.com/file/d/1y8gjAvnt0kkf1dSEyauVFu2weLi53LmF/view?usp=drive_link) / [Baidu Netdisk](https://pan.baidu.com/s/1jr01T_hzl8K_h2VksrmlFQ?pwd=97lm) (97lm) 
| DID-Data | [Google Drive](https://drive.google.com/file/d/1RDvMFZn57UFrkeeojRHXwR7YbvXSGR5i/view?usp=drive_link) / [Baidu Netdisk](https://pan.baidu.com/s/1PJrRTDsG4vL4XwhNd8kfHg?pwd=5g4p) (5g4p) 
| DDN-Data | [Google Drive](https://drive.google.com/file/d/1p7FVQuZSw4n0nXEvLrsJPtYxzlMyOCK0/view?usp=drive_link) / [Baidu Netdisk](https://pan.baidu.com/s/1Y3YRkNO40m6bII-R3-Hi4g?pwd=b0b5) (b0b5) 
| SPA-Data | [Google Drive](https://drive.google.com/file/d/1hEpYFrFG0qhKassfYAZmXwUnNUYmGMLs/view?usp=drive_link) / [Baidu Netdisk](https://pan.baidu.com/s/1CO7wlaZyhu2egjfdaavFeQ?pwd=x0i5) (x0i5) 


## 🚨 Performance Evaluation
See folder "evaluations" 

1) *for Rain200L/H and SPA-Data datasets*: 
PSNR and SSIM results are computed by using this [Matlab Code](https://github.com/sauchm/MSDT/tree/main/evaluations/Evalution_Rain200L_Rain200H_SPA-Data).

2) *for DID-Data and DDN-Data datasets*: 
PSNR and SSIM results are computed by using this [Matlab Code](https://github.com/sauchm/MSDT/tree/main/evaluations/Evaluation_DID-Data_DDN-Data).



## 🚀 Visual Deraining Results

| Methods | MSDT |
|:-----: |:-----: |
| Rain200L | [Baidu Netdisk](https://pan.baidu.com/s/1us3smvwhAe3azJPnunWs8w?pwd=1xkc) (1xkc) 
| Rain200H | [Baidu Netdisk](https://pan.baidu.com/s/1S__NNB0jV2ING2ngR0PjiA?pwd=yr3n) (yr3n) 
| DID-Data | [Baidu Netdisk](https://pan.baidu.com/s/1Rif4QC1AuDF4ccHteg_A4A?pwd=242e) (242e) 
| DDN-Data | [Baidu Netdisk](https://pan.baidu.com/s/1JFHyrTMSdsFotOJ6pKokow?pwd=2pwk) (2pwk) 
| SPA-Data | [Baidu Netdisk](https://pan.baidu.com/s/14fSFf_T7AOD44ktso56Rxw?pwd=cag0) (cag0) 


## 👍 Acknowledgement
Thanks for their awesome works ([DeepRFT](https://github.com/INVOKERer/DeepRFT) and [DRSformer](https://github.com/cschenxiang/DRSformer)).

## 📘 Citation
Please consider citing our work as follows if it is helpful.
```
@inproceedings{chen2024rethinking,
  title={Rethinking Multi-Scale Representations in Deep Deraining Transformer},
  author={Chen, Hongming and Chen, Xiang and Lu, Jiyang and Li, Yufeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={2},
  pages={1046--1053},
  year={2024}
}
```

