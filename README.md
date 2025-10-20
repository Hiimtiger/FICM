# FICM: Force Informed Cell Map 
FICM: Force Informed Cell Map Image Synthesis using Attention-gated Dual Encoder U-Net for Carcinoma Cell Line Classification    

- Carcinoma Cell Line Classification Problem: **HCC827** & **A549** are two types of lungadenocarcinomacell lines that display different aggressiveness behaviors. Though they can be separated via staining, different staining methods may influence their actual behaviors. Furthermore, cellular morphology can be completely stochastic, **making it very challenging and highly subjective to separate them via fluorescence signals using the naked eye.**
- In our research, we propose a **new medical image modality** that combined a cell's morphological feature with its internal force gradients captured with force-sensing chips. Our prroposed modality is able to enhance classification accuracy by 37.67% compared to unprocessed modalities.

## Modality Design: 
The proposed fusion modality combines two major features of a cell, the boundary feature and its internal force gradient. 
<p align="center">
  <img src="./assets_ficm/p2.gif" width="600" alt="Figure 1">
  <br>
  <b>Figure 1.</b> Overview of the proposed model architecture.
</p>

## Modality fusion Model Architecture:
Our model architecture is inspired by [MRI-Styled PET](https://ieeexplore.ieee.org/document/10918787). We adpoted the dual-encoder structure to extract features from both fluorescence signals and force gradient. We added a Squeeze-and-Excitation Module to adapt the fusion weights channel-by-channel. Our loss function consists of equal weights of dice loss and smooth L1 loss. 

## Classification model architecture and classification results:
