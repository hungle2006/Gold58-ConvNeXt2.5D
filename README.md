<img width="1423" height="942" alt="image" src="https://github.com/user-attachments/assets/0f9b89d2-83a4-4382-8c0a-f00ea541891a" />

---

# KneeGold-B0
## Fixed-Gold 2.5D ConvNeXtV2 Baseline for Multi-Label Knee MRI Classification

---

## 1. Overview

**KneeGold-B0** is a study-level deep learning baseline for predicting 12 knee MRI abnormalities.

Main characteristics:

- 4,349 studies are used for training.
- 58 studies with official labels are reserved as a fixed validation set.
- Validation studies are never used for training.
- MRI series are organized into five protocol-aware slots.
- Slices are sampled uniformly from 6% to 94% of each selected series.
- Three neighboring slices are stacked to form a 2.5D image.
- ConvNeXtV2-Tiny is used as the image encoder.
- Mean pooling and max pooling are used to aggregate clip-level features.
- The model outputs 12 independent probabilities.
- Training is performed on two NVIDIA T4 GPUs using Distributed Data Parallel.

---

## 2. Prediction Targets

The model predicts the following 12 abnormalities:

1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's
11. Contusion
12. Fracture

This is a multi-label classification problem.

Each MRI study produces 12 independent output probabilities.

---

## 3. Train and Validation Split

The dataset contains 4,407 pseudo/manual-labeled studies.

Among them, 58 studies contain complete official ground-truth labels.

These 58 studies are permanently excluded from training.

```text
4,407 pseudo/manual-labeled studies
                |
                | remove 58 official Gold studies
                |
        ---------------------
        |                   |
    4,349 TRAIN         58 VALIDATION
        |                   |
 pseudo/manual labels    official labels
        |                   |
 backpropagation         evaluation only
```

Important rule:

```text
TRAIN StudyInstanceUID intersection VALID StudyInstanceUID = 0
```

Therefore, no study-level leakage occurs between training and validation.

---

## 4. Why Use the 58 Gold Studies for Validation?

A random train-validation split would mainly measure performance against pseudo/manual labels.

Instead, KneeGold-B0 uses the 58 official studies as a fixed validation set.

This has three advantages:

- The validation labels are more reliable.
- Every future experiment is evaluated on exactly the same studies.
- Model improvements can be compared fairly.

The main validation metric is macro ROC-AUC.

---

## 5. MRI Study Structure

One MRI study may contain several MRI series.

Example:

```text
StudyInstanceUID
|
|-- Sagittal Series
|   |-- Slice 1
|   |-- Slice 2
|   |-- Slice 3
|   `-- ...
|
|-- Coronal Series
|   |-- Slice 1
|   |-- Slice 2
|   `-- ...
|
`-- Axial Series
    |-- Slice 1
    |-- Slice 2
    `-- ...
```

The model treats one StudyInstanceUID as one training sample.

Slices and series from the same study are never split across train and validation.

---

## 6. Reusing the Existing MRI Cache

The training pipeline directly reuses the preprocessed cache.

Expected structure:

```text
rsna-knee-dicom-cache/
|
|-- index/
|   `-- train_series_index.pkl
|
`-- processed/
    `-- train_series/
        `-- StudyInstanceUID/
            |-- SeriesInstanceUID.npy
            `-- SeriesInstanceUID.pkl
```

The cache stores MRI slices in uint8 format.

The original standardized range was approximately:

```text
-5 to 5
```

and was encoded into:

```text
0 to 255
```

During training, the standardized values are restored using:

```python
x = x / 255.0 * 10.0 - 5.0
```

This allows the model to reuse the cached MRI data without decoding all DICOM files again.

---

## 7. Protocol-Aware Series Selection

For each MRI study, the baseline selects up to five series slots.

| Slot | Plane | Sequence Type | Number of Center Slices |
|---|---|---|---:|
| SAG-FLUID | Sagittal | Fluid-sensitive | 18 |
| SAG-NONFLUID | Sagittal | Non-fluid-sensitive | 14 |
| COR-FLUID | Coronal | Fluid-sensitive | 12 |
| COR-NONFLUID | Coronal | Non-fluid-sensitive | 8 |
| AXIAL | Axial | Any | 12 |
| Total maximum | - | - | 64 |

The selected series are chosen using their MRI metadata.

If several series match the same slot, the series with the largest number of slices is preferred.

Fat suppression is used only as a secondary preference.

A series is not duplicated across multiple slots.

---

## 8. Slice Sampling Strategy

Slices are not taken only from the center.

Instead, the model samples each selected MRI series uniformly between 6% and 94% of the ordered slice stack.

For a series containing `N` slices:

```text
first_index = 0.06 * (N - 1)
last_index  = 0.94 * (N - 1)
```

If the selected slot requires `K` center slices, the center positions are distributed approximately uniformly between `first_index` and `last_index`.

Example with 100 slices:

```text
N = 100

first_index = 0.06 * 99
            = 5.94
            = approximately slice 6

last_index  = 0.94 * 99
            = 93.06
            = approximately slice 93
```

Therefore, the useful sampling region is approximately:

```text
slice 6 ------------------------------------------ slice 93
   |                                                   |
   |<-------------- uniform sampling ----------------->|
```

If 18 center slices are required:

```text
6 -- 11 -- 16 -- 21 -- 26 -- ... -- 83 -- 88 -- 93
```

This strategy provides broad anatomical coverage while avoiding the extreme boundary slices.

------ sampling ------->   |
                  6%                          94%
```

If 18 slices are required, 18 center positions are distributed approximately uniformly inside this region.

Example:

```text
6 ---- 11 ---- 16 ---- 21 ---- 26 ---- ... ---- 83 ---- 88 ---- 93
^       ^       ^       ^       ^                 ^       ^       ^
sample  sample  sample  sample  sample            sample  sample  sample
```

This strategy provides broad anatomical coverage while avoiding extreme boundary slices.

---

## 9. 2.5D Image Construction

Each selected center slice is converted into a 2.5D image.

For center slice `i`:

```text
Channel 1 = slice i-1
Channel 2 = slice i
Channel 3 = slice i+1
```

Example:

```text
Slice 33 ----Slice 34 ------> 2.5D image [3, 224, 224]
Slice 35 ----/
```

The three slices are used as the three channels of one input image.

This gives the model local depth information while still allowing the use of a 2D CNN backbone.

---

## 10. Maximum Number of 2.5D Clips

Maximum number of center slices per study:

```text
Sagittal fluid-sensitive     = 18
Sagittal non-fluid           = 14
Coronal fluid-sensitive      = 12
Coronal non-fluid            = 8
Axial                        = 12
----------------------------------
Maximum                      = 64
```

Maximum number of 2.5D clips used for training in one epoch:

```text
number_of_training_studies = 4,349
maximum_clips_per_study    = 64

maximum_clips_per_epoch
= 4,349 * 64
= 278,336
```

Therefore, the theoretical maximum is:

```text
278,336 2.5D clips per epoch
```

The actual number can be lower because some studies may not contain all five MRI slots.

The 58 Gold validation studies are not included in this training count.

----------------------------------
Maximum                      = 64
```

Therefore:

```text
4,349 training studies
x
64 maximum clips per study
=
278,336 maximum 2.5D clips per epoch
```

The real number may be slightly lower because some studies may not contain all five required MRI slots.

The 58 Gold validation studies are not included in this training count.

---

## 11. Data Augmentation

Training augmentation is applied to the complete 3-channel 2.5D clip.

The same spatial transformation is applied to all three channels.

The baseline uses conservative augmentation:

- Rotation up to approximately 7 degrees
- Small translation
- Scale between 0.95 and 1.05
- Mild brightness variation
- Mild contrast variation
- Small Gaussian noise

Horizontal flipping is not used.

Reason:

```text
Horizontal flipping may alter medial and lateral anatomical semantics.
```

---

## 12. ConvNeXtV2-Tiny Encoder

Each 2.5D image is encoded by a pretrained ConvNeXtV2-Tiny backbone.

Backbone:

```text
convnextv2_tiny.fcmae_ft_in22k_in1k
```

Input:

```text
[3, 224, 224]
```

Processing:

```text
2.5D MRI clip
      |
      v
ConvNeXtV2-Tiny
      |
      v
clip-level feature vector
```

The backbone is initialized with pretrained weights and fine-tuned on the knee MRI task.

---

## 13. GPU Memory Strategy

A complete study may contain dozens of 2.5D clips.

To avoid moving every clip to the GPU at the same time, the encoder processes the clips in smaller chunks.

Example:

```text
Study with 64 clips
        |
        | split into GPU chunks
        v
Clip 1-32
Clip 33-64
        |
        v
ConvNeXtV2
        |
        v
64 feature vectors
```

This reduces peak GPU memory usage.

---

## 14. Study-Level Feature Aggregation

After ConvNeXtV2 encoding, a study contains multiple feature vectors:

```text
f1, f2, f3, ..., fM
```

KneeGold-B0 uses two simple pooling strategies.

### Mean Pooling

Mean pooling captures information distributed across the MRI study.

```text
f1
f2
f3
...
fM
 |
 v
MEAN
 |
 v
mean feature
```

### Max Pooling

Max pooling preserves strong localized activations.

```text
f1
f2
f3
...
fM
 |
 v
MAX
 |
 v
max feature
```

The two vectors are concatenated.

```text
mean feature ----                  +--> concatenated study feature
max feature -----/
```

---

## 15. Classification Head

The final study representation is passed through:

```text
Mean feature
      +
Max feature
      |
      v
Concatenate
      |
      v
LayerNorm
      |
      v
Dropout
      |
      v
Linear layer
      |
      v
12 logits
      |
      v
Sigmoid
      |
      v
12 probabilities
```

No label-aware attention is used in this baseline.

No label Transformer is used.

No 3D Transformer is used.

The purpose is to keep the baseline simple and interpretable.

---

## 16. Loss Function

The model uses weighted Binary Cross-Entropy with logits.

Class imbalance is handled using a positive-class weight calculated from the training set.

```text
positive_weight
= number_of_negative_samples / number_of_positive_samples
```

Example:

```text
negative samples = 3,000
positive samples = 1,000

positive_weight
= 3,000 / 1,000
= 3.0
```

The weight is clipped to prevent extremely large values.

Missing labels are ignored by the loss.

---

## 17. Optimizer and Learning Rate

Optimizer:

```text
AdamW
```

Two learning rates are used.

| Component | Learning Rate |
|---|---:|
| ConvNeXtV2 backbone | 0.00002 |
| Classification head | 0.0002 |

Additional settings:

```text
Weight decay        = 0.0001
Gradient clipping   = 1.0
Warm-up             = 1 epoch
Scheduler           = cosine decay
Mixed precision     = enabled
```

---

## 18. Dual-T4 Training

Training is designed for two NVIDIA T4 GPUs using PyTorch Distributed Data Parallel.

```text
                  4,349 training studies
                           |
                           v
                  DistributedSampler
                           |
               +-----------+-----------+
               |                       |
               v                       v
           T4 GPU 0                T4 GPU 1
            Rank 0                  Rank 1
               |                       |
               v                       v
          ConvNeXtV2               ConvNeXtV2
               |                       |
               +------ gradient -------+
                      synchronization
                           |
                           v
                  synchronized model
```

Each GPU processes a different subset of studies.

The gradients are synchronized using NCCL.

---

## 19. Effective Batch Size

The baseline configuration uses:

```text
batch_size_per_gpu    = 2 studies
number_of_gpus        = 2
gradient_accumulation = 2
```

The effective batch size is:

```text
effective_batch_size
= batch_size_per_gpu
  * number_of_gpus
  * gradient_accumulation

= 2 * 2 * 2

= 8 studies per optimizer update
```

Mixed precision is used to improve training speed and reduce GPU memory usage.

---

## 20. Gold58 Validation

After each epoch, the model is evaluated on the same 58 Gold studies.

Validation performs only forward inference.

No gradient update is performed.

For each abnormality:

```text
Prediction probability
        |
        v
ROC-AUC
```

The overall validation score is the mean of the valid per-label ROC-AUC scores.

This is reported as:

```text
Gold58 Macro ROC-AUC
```

The checkpoint with the best Gold58 Macro ROC-AUC is saved.

---

## 21. Complete KneeGold-B0 Pipeline

```text
                 4,407 pseudo/manual labels
                           |
                           | remove Gold58
                           v
                  4,349 TRAIN STUDIES
                           |
                           v
                 Existing MRI cache
                           |
                           v
                Select five MRI slots
                           |
                           v
                Uniform slice sampling
                     from 6% to 94%
                           |
                           v
              Build 2.5D three-slice clips
                  [i-1, i, i+1]
                           |
                           v
                  ConvNeXtV2-Tiny
                           |
                           v
                 Clip-level features
                           |
                +----------+----------+
                |                     |
                v                     v
          Mean pooling           Max pooling
                |                     |
                +----------+----------+
                           |
                           v
                     Concatenate
                           |
                           v
                      LayerNorm
                           |
                           v
                        Dropout
                           |
                           v
                     Linear(12)
                           |
                           v
                     12 logits
                           |
                           v
                 Weighted BCE loss
                           |
                           v
                    DDP on T4 x2
                           |
                           v
             Evaluate on 58 Gold studies
                           |
                           v
                 Gold58 Macro ROC-AUC
```

---

## 22. Why This Model Is a Baseline

KneeGold-B0 intentionally does not include:

- Label-aware attention
- Gated attention
- Label Transformer
- Depth Transformer
- GeM pooling
- Learned focal slice selection
- CoAtNet
- DINOv3
- VideoMAE
- 3D CNN
- Multi-model ensemble

This allows every future improvement to be evaluated independently.

---

## 23. Planned Model Development

Future experiments can be organized as follows:

```text
KneeGold-B0
ConvNeXtV2 + uniform 2.5D + Mean/Max pooling
        |
        v
KneeGold-B1
+ label-aware attention
        |
        v
KneeGold-B2
+ DINOv3 representation
        |
        v
KneeGold-B3
+ global and focal slice sampling
        |
        v
KneeGold-B4
+ CoAtNet branch
        |
        v
KneeGold-B5
+ 3D VideoMAE or Video Swin branch
        |
        v
Final Ensemble
```

All experiments should use the same 58 Gold studies for validation.

---

## 24. Main Advantages

### Reliable Validation

The model is selected using 58 official ground-truth studies instead of a random pseudo-label split.

### Anatomical Coverage

Sagittal, coronal, and axial MRI views are explicitly represented.

### Sequence Diversity

Fluid-sensitive and non-fluid-sensitive series are both included.

### Efficient Depth Context

2.5D clips provide neighboring-slice information without the memory cost of a full 3D network.

### Simple Study Aggregation

Mean and max pooling provide a strong, transparent baseline.

### Efficient Training

The existing MRI cache, mixed precision, chunked image encoding, and dual-T4 DDP reduce computational overhead.

---

## 25. Short Presentation Script

> We propose KneeGold-B0 as a study-level baseline for multi-label knee MRI classification.
>
> The key design choice is the validation protocol. We reserve all 58 studies with complete official labels as a fixed Gold validation set. These studies are completely excluded from optimization. The remaining 4,349 pseudo- or manually labeled studies are used for training.
>
> For each MRI study, we select up to five protocol-aware MRI series covering sagittal, coronal, and axial views. We then uniformly sample center slices between 6 and 94 percent of each selected volume. Depending on the MRI protocol, we select 18 sagittal fluid-sensitive centers, 14 sagittal non-fluid centers, 12 coronal fluid-sensitive centers, 8 coronal non-fluid centers, and 12 axial centers.
>
> Each center slice is converted into a 2.5D input by stacking the previous, current, and next MRI slices as three channels. These clips are encoded using a pretrained ConvNeXtV2-Tiny backbone.
>
> At the study level, all clip features are aggregated using both mean pooling and max pooling. The pooled representations are concatenated and passed to a linear classification head that predicts 12 knee abnormalities.
>
> Training uses weighted binary cross-entropy, AdamW optimization, cosine learning-rate decay, mixed precision, gradient accumulation, and Distributed Data Parallel on two NVIDIA T4 GPUs.
>
> After every epoch, the model is evaluated on the fixed 58-study Gold validation set using macro ROC-AUC. The checkpoint with the best Gold58 score is retained.
>
> We intentionally keep this architecture simple so that more advanced components, including label-aware attention, DINOv3, learned focal slice selection, CoAtNet, and 3D VideoMAE, can be added later and evaluated through controlled ablation studies.

---

## 26. Model Name

**KneeGold-B0**

Full name:

**KneeGold-B0: Fixed-Gold 2.5D ConvNeXtV2 Mean-Max MIL Baseline**

