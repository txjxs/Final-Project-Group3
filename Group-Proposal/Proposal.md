# Project Proposal: Conditional GAN for On-Demand Shoe Generation

**Date:** November 7, 2025

**Status:** Draft

---

##  Team Members

* Tejas Nisar
* Jasreen Kaur Mehta
* Mohammed Ismail Sarfaraz Shaik

---

##  Project Overview

This project aims to implement, train, and **critically compare several distinct Conditional Generative Adversarial Network (cGAN) architectures**. The objective is to determine the most effective and stable architecture for generating high-fidelity, controllable synthetic images.

All models will be trained on the **UT Zappos 50k shoe dataset**.

The final output will be a comprehensive research report comparing the models, and an interactive web application built with Streamlit to deploy the **single best-performing model** from our experiment.

---

##  Key Objectives

1.  **Implement:** Successfully build several distinct, functional, and stable cGAN architectures that explore different approaches to image conditioning.
2.  **Train:** Train all models on an *identical*, pre-processed subset of the Zappos 50k dataset to ensure a fair and direct comparison.
3.  **Compare & Evaluate:** Rigorously evaluate the models on training stability, visual fidelity (realism), diversity, and adherence to the conditional prompt. This comparison will use both qualitative and quantitative metrics.
4.  **Demo:** Develop an accessible Streamlit app to deploy the **winning** model, allowing users to select a shoe category and receive a generated image.

---

##  Project Phases

Our workflow is divided into four main phases:

### Phase 1: Foundation & Data Preparation
* **Data:** The team will acquire, clean, and establish a unified preprocessing pipeline for the UT Zappos 50k dataset. This includes image resizing, normalization, and extracting/formatting the conditional labels (e.g., shoe categories) to ensure all models are trained on the exact same data.
* **Baseline:** The team will agree on a common base architecture to build from.

### Phase 2: Comparative Model Development
* **Implementation:** The team will implement several different cGAN variants. These models will explore different conditioning mechanisms (how the label is fed into the network) and training objectives.
* **Training:** All implemented models will be trained systematically on the prepared dataset. Training logs and model checkpoints will be saved for the evaluation phase.

### Phase 3: Evaluation & Selection
* **Analysis:** The team will conduct a head-to-head comparison of all trained models. This involves generating comparison grids and calculating quantitative metrics for each.
* **Selection:** Based on the evidence, the team will select the "champion" model that provides the best balance of image quality, stability, and prompt adherence.

### Phase 4: Demo Deployment
* **Interface:** Using the champion model from Phase 3, the team will build the final Streamlit web application.
* **Integration:** The pre-trained generator model will be loaded into the app to create the final user-facing demo that allows users to select a shoe category and receive a generated image.

---

##  Dataset

* **Primary Dataset:** UT Zappos 50k (Zappos50k)
* **Description:** A large-scale shoe dataset with over 50,000 catalog images. Crucially, it includes rich metadata (e.g., categories, materials, styles) which will serve as the "conditions" for our cGAN.
* **Source:** [https://vision.cs.utexas.edu/projects/finegrained/utzap50k/](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/)
* **Citations:** 
    * A. Yu and K. Grauman. "Fine-Grained Visual Comparisons with Local Learning". In CVPR, 2014.
    * A. Yu and K. Grauman. "Semantic Jitter: Dense Supervision for Visual Comparisons via Synthetic Images". In ICCV, 2017.

---
