# Spatiotemporal Deepfake Video Detection using Face-Centric Representations

## Overview

This project investigates **spatiotemporal deepfake video detection** using **3D Convolutional Neural Networks (3D CNNs)**. As modern deepfakes exhibit fewer frame-level artifacts, the project focuses on learning **temporal inconsistencies across video frames** rather than relying on isolated images.

The core objective is to **compare full-frame vs. face-centric spatiotemporal representations** under a **uniform frame sampling strategy**, and evaluate which approach yields more robust and generalizable deepfake detection performance.

---

## Key Contributions

* Implemented a **3D ResNet-18–based deepfake detection pipeline** for spatiotemporal video analysis
* Designed **uniform frame sampling (16 evenly spaced frames)** across entire videos
* Built and compared:

  * **Full-frame spatiotemporal model**
  * **Face-centric spatiotemporal model** using **MTCNN-based facial cropping**
* Conducted controlled experiments on **DFDC** and **FaceForensics++** datasets
* Demonstrated that **face-centered representations improve robustness and reduce overfitting**

---

## Datasets

* **DeepFake Detection Challenge (DFDC)**
* **FaceForensics++**

Each video is processed into fixed-length clips of **16 uniformly sampled frames**, enabling broad temporal coverage while maintaining consistent input size.

---

## Methodology

### 1. Frame Sampling

* Uniformly sampled **16 evenly spaced frames** across the full video duration
* Ensures broad temporal coverage without continuous sampling

### 2. Preprocessing Pipelines

Two parallel datasets were constructed:

**Full-Frame Pipeline**

* Uses raw video frames
* Resized to **112×112**
* Normalized using ImageNet statistics

**Face-Centric Pipeline**

* Applies **MTCNN** for face detection
* Crops facial regions from each sampled frame
* Focuses the model on manipulation-prone regions

---

## Model Architecture

* **Backbone:** 3D ResNet-18
* **Pretraining:** Kinetics-400
* **Input:** 16-frame spatiotemporal clips
* **Task:** Binary classification (Real vs Fake)

3D convolutions enable joint modeling of **spatial and temporal features**, capturing artifacts such as flickering, unnatural motion, and inconsistent facial textures.

---

## Training Setup

* **Framework:** PyTorch
* **Optimizer:** Adam
* **Learning Rate:** 1e-4
* **Loss Function:** Binary Cross-Entropy with Logits
* **Batch Size:** 4
* **Early Stopping:** Based on validation AUC
* **Scheduler:** ReduceLROnPlateau

Models were trained independently for full-frame and face-centric inputs.

---

## Results

| Model Type           | Accuracy | F1-score  | AUC       |
| -------------------- | -------- | --------- | --------- |
| Full-frame           | ~95%     | ~0.94     | ~0.979    |
| Face-centric (MTCNN) | **~96%** | **~0.95** | **~0.99** |

### Key Observations

* Face-centric models showed **more stable convergence**
* Reduced sensitivity to background noise and identity leakage
* Improved generalization compared to full-frame inputs

---

## Discussion

The results demonstrate that **face-centered spatiotemporal representations** are more discriminative than full-frame inputs when using uniform frame sampling. By isolating facial regions—where manipulations primarily occur—the model avoids learning spurious correlations from background or identity-specific cues.

---

## Limitations

* Only **uniform frame sampling** was evaluated
* No **textual metadata or multimodal fusion** included
* Potential **identity overlap** across dataset splits may inflate performance

---

## Future Work

* Compare **continuous vs non-continuous frame sampling strategies**
* Integrate **textual metadata** for multimodal deepfake detection
* Enforce **identity-disjoint dataset splits** to better evaluate generalization
* Explore transformer-based temporal modeling for long-range dependencies

---

## Technologies Used

* Python
* PyTorch
* 3D ResNet-18
* MTCNN
* Computer Vision
* Deep Learning
* Video Forensics

---

## Authors

* **Bhavishya Vudatha**
* **Sai Krishna Karnam**

---

## References

* de Lima et al., *Deepfake Detection Using Spatiotemporal Convolutional Networks*, ICASSP 2020
* Tariq et al., *A Convolutional LSTM-Based Residual Network for Deepfake Video Detection*, 2020
* Tung & Tung, *DeepSneak: Deepfake Video Detection*, 2024
* DFDC Dataset, Facebook AI
* FaceForensics++, Rössler et al.
