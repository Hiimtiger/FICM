# FICM: Force Informed Cell Map 
FICM: Force Informed Cell Map Image Synthesis using Attention-gated Dual Encoder U-Net for Carcinoma Cell Line Classification    

- Carcinoma Cell Line Classification Problem: **HCC827** & **A549** are two types of lungadenocarcinomacell lines that display different aggressiveness behaviors. Though they can be separated via staining, different staining methods may influence their actual behaviors. Furthermore, cellular morphology can be completely stochastic, **making it very challenging and highly subjective to separate them via fluorescence signals using the naked eye.**
- In our research, we propose a **new medical image modality that combined a cell's morphological feature with its internal force gradients captured with force-sensing chips**. Our prroposed modality is able to enhance classification accuracy by 37.67% compared to unprocessed modalities.
<p align="center">
  <img src="./assets_ficm/poster_montage.png" width="1000" alt="Figure 1">
  <br>
  <b>Figure 1.</b> Montage of our proposed modality compared with original modalities.
</p>

## Modality Design: 
The proposed fusion modality combines two major features of a cell, the boundary feature and its internal force gradient. (See **Figure 2.**)
<p align="center">
  <img src="./assets_ficm/p2.gif" width="600" alt="Figure 2">
  <br>
  <b>Figure 2.</b> Overview of the proposed model architecture.
</p>

Our training method allowed complementary modality fusion, which would **use both features from fluorecence signals and its force gradient to decide** the most accurate cell boundary while revealing its instantaneous internal force gradients. (See **Figure 3.** and **Figure 4.**)
<table align="center">
  <tr>
    <td align="center">
      <img src="./assets_ficm/p3.png" width="500" alt="Figure 3"><br>
      <b>Figure 3.</b> Case when lacking fluorescence signal.
    </td>
    <td align="center">
      <img src="./assets_ficm/p4.png" width="500" alt="Figure 4"><br>
      <b>Figure 4.</b> Case when lacking force gradient signals.
    </td>
  </tr>
</table>

## Modality fusion Model Architecture:
Our model architecture is inspired by [MRI-Styled PET](https://ieeexplore.ieee.org/document/10918787). We adpoted a dual-encoder structure to extract features from both fluorescence signals and force gradient. We added a Squeeze-and-Excitation Module at the bottleneck to adapt the fusion weights channel-by-channel. Our loss function consists of equal weights of dice loss and masked smooth L1 loss (only consider the accumulated loss within the boundary). 
<p align="center">
  <img src="./assets_ficm/model_architecture.svg" width="600" alt="Figure 5">
  <br>
  <b>Figure 5.</b> Overview of the proposed dual-encoder modlaity fusion model architecture.
</p>

## Classification model architecture and classification results:
We used a sinmple 4-layer CNN model for this classification task.
<p align="center">
  <img src="./assets_ficm/classification_model_architecture.svg" width="600" alt="Figure 6">
  <br>
  <b>Figure 6.</b> Overview of our classification model architecture.
</p>

**Figure 7~10** are confusion matrices of different modalitites using the same test dataset.
<table align="center">
  <tr>
    <td align="center">
      <img src="./assets_ficm/pure_fluor_confusion_matrix.png" width="250" alt="Figure 7"><br>
      <b>Figure 7.</b> Pure fluorescence images
    </td>
    <td align="center">
      <img src="./assets_ficm/pure_grid_confusion_matrix.png" width="250" alt="Figure 8"><br>
      <b>Figure 8.</b> Pure force grid images
    </td>
    <td align="center">
      <img src="./assets_ficm/fluorandgrid_confusion_matrix.png" width="200" alt="Figure 9"><br>
      <b>Figure 9.</b> Stacked fluorescence and force grid
    </td>
    <td align="center">
      <img src="./assets_ficm/FICM_confusion_matrix.png" width="250" alt="Figure 10"><br>
      <b>Figure 10.</b> Our proposed fusion modality.
    </td>
  </tr>
</table>

