---
title: CIFAR-10 Dense MLP vs MobileNetV2 CNN
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.25.0
app_file: app.py
pinned: false
license: mit
---

# CIFAR-10: Dense MLP vs. Fine-Tuned MobileNetV2 CNN

Interactive demonstration comparing a **Pure Dense Neural Network (MLP)** against a **Fine-Tuned MobileNetV2 CNN** on CIFAR-10.

Built as part of the Deep Learning Capstone Project at **Tuwaiq Academy** by **Khalid & Wasan**.

## Overview

- **Dense Baseline (Shipped Model)**: 3 × 512 Dense layers, Dropout 0.3, Adam ($lr=10^{-4}$). Test accuracy: **54.08%**. Evaluates flattened 1D vectors ($3,072$ features) where pixel adjacency is destroyed.
- **MobileNetV2 CNN (Fine-Tuned)**: Depthwise separable convolutions pre-trained on ImageNet and fine-tuned in only 4 epochs. Test accuracy: **~80%+**. Preserves 2D spatial arrangement, local textures, and translation equivariance.

## How to Test
1. Select any sample image from the pre-loaded CIFAR-10 gallery (representing all 10 classes) or upload your own 32×32 or standard resolution image.
2. Click **Run Dual Model Inference** to see predictions, confidence bars, latency, and spatial explanations side-by-side.
