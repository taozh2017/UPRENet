# FSMIS via UPRENet

![image](https://github.com/zmcheng9/GMRD/blob/main/overview.png)

### Abstract
Few-Shot Learning (FSL) has garnered increasing attention for data-scarce scenarios, particularly in medical segmentation tasks where only a few labeled data points are available. Existing few-shot segmentation methods typically learn prototypes from support images and employ nearest-neighbor searching to segment query images. Despite notable progress, effectively learning prototypes for each class remains a challenging task to achieve promising results. In this paper, we propose an Uncertainty-guided Prototype Reliability Enhancement Network (UPRE-Net) for few-shot medical image segmentation. Specifically, we present a dual-support branch to maximize the extraction of information from support images through augmentation techniques. To enhance the reliability of prototypes, we propose an Uncertainty-guided Prototype Generation (UPG) module. Within the UPG module, we first extract both global and local prototypes for each class and then apply uncertainty measures to select the most informative prototypes. Additionally, to effectively combine the prediction results from the dual-support branch, we present a Reliable Dynamic Fusion (RDF) module. This module dynamically integrates the two prediction results to generate a more reliable output. Furthermore, we present an Uncertainty-induced Weighted Loss (UWL) to ensure that the model pays more attention to these regions with high uncertainty. Experiments on four benchmark medical image datasets demonstrate that our proposed model significantly outperforms state-of-the-art methods. 

### Dependencies
Please install following essential dependencies:
```
dcm2nii
json5==0.9.6
jupyter==1.0.0
nibabel==5.2.1
numpy==1.26.4
opencv-python==4.10.0.84
Pillow>=8.1.1
sacred==0.8.5
scikit-image==0.23.2
SimpleITK==2.3.1
torch==2.0.1
torchvision=0.15.2
tqdm==4.66.4
```

### Data sets and pre-processing
Download:
1) **CHAOS-MRI**: [Combined Healthy Abdominal Organ Segmentation data set](https://chaos.grand-challenge.org/)
2) **Synapse-CT**: [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)
3) **CMR**: [Multi-sequence Cardiac MRI Segmentation data set](https://zmiclab.github.io/projects/mscmrseg19/) (bSSFP fold)

Pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their github repository.

### Training
1. Compile `./data/supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./data/supervoxels/setup.py build_ext --inplace`) and run `./data/supervoxels/generate_supervoxels.py` 
2. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.  
3. Run `./script/train.sh` 

### Inference
Run `./script/evaluate.sh` 

### Citation
```
@ARTICLE{11203259,
  author={Hu, Junfei and Zhou, Tao and Huang, Kaiwen and Zhou, Yi and Zhang, Haofeng and Fan, Boqiang and Fu, Huazhu},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Uncertainty-guided Prototype Reliability Enhancement Network for Few-Shot Medical Image Segmentation}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Image segmentation;Prototypes;Biomedical imaging;Uncertainty;Reliability;Training;Feature extraction;Resource description framework;Few shot learning;Annotations;Few-shot learning;medical image segmentation;prototype enhancement;reliable dynamic fusion},
  doi={10.1109/TMI.2025.3621452}}
```
