# AI of GOD 4.0

This repository contains deep learning solutions developed for the **[AI-OF-GOD-4 Kaggle Competition](https://www.kaggle.com/competitions/AI-OF-GOD-4)**.  
The project focuses on **image-based classification** using **Convolutional Neural Networks (CNNs)** and **transfer learning** techniques to predict class labels for the provided dataset.

---

## Project Overview

The competition involves classifying images from a custom dataset using robust and efficient neural network architectures.  
Three main models are implemented and compared for performance and generalization:

1. **ConvNeXt-based Model** (`homoscedasticity.py`)  
   - Uses transfer learning with pretrained **ConvNeXt Base** backbone.  
   - Implements advanced data augmentation (ResizedCrop, ColorJitter, Rotation, TrivialAugmentWide).  
   - Handles class imbalance through **weighted loss computation**.  
   - Trains with **AdamW optimizer** and **Cosine Annealing learning rate scheduling**.  
   - Generates predictions and produces a formatted `submission.csv` file.

2. **Custom ResNet-50 Model** (`homoscedasticity_resnet_50.py`)  
   - Implements a **ResNet architecture from scratch** using residual connections.  
   - Includes custom dataset loaders for training and testing.  
   - Trains a deep CNN with **CrossEntropyLoss** and **Adam optimizer**.  
   - Outputs class probabilities and final prediction CSVs for Kaggle submission.

3. **EfficientNet-B3 Model** (`homoscedasticity_efficientnet_b3.py`)  
   - Fine-tunes pretrained **EfficientNet-B3** layers on the competition dataset.  
   - Uses **layer-wise learning rates** for stable optimization.  
   - Applies **weighted loss functions** for handling unbalanced data.  
   - Utilizes **Cosine Annealing scheduler** for adaptive learning rate control.  
   - Produces Kaggle-ready predictions in `submission.csv`.

---

## Key Features

- **Data Preprocessing & Augmentation:** Strong image transformations to improve model robustness.  
- **Modeling:** Combines pretrained architectures (ConvNeXt, EfficientNet-B3) and a custom-built ResNet-50.  
- **Optimization:** Includes warmup, cosine scheduling, and mixed-precision training.  
- **Class Imbalance Handling:** Weighted loss computation for fair class learning.  
- **Complete Pipeline:** Data loading, model training, validation, and submission generation are fully automated.

---

### Results

- **ConvNeXt-based Model Accuracy:** 93.1666%  
- **EfficientNet-B3 Model Accuracy:** 87.3170%  
- **ResNet-50-based Model Accuracy:** 73.1196%  

---

## Tech Stack

- **Language:** Python  
- **Framework:** PyTorch, Torchvision  
- **Libraries:** NumPy, Pandas, PIL, tqdm, scikit-learn   

---

## Dataset

The dataset used in this project is provided by the **AI-OF-GOD-4 Kaggle competition**.  
You can access it directly here:  
🔗 [AI-OF-GOD-4 Dataset](https://www.kaggle.com/competitions/AI-OF-GOD-4)

---
