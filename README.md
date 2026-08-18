# cnn-handwritten-character-classification
CNN-based handwritten character classification using EMNIST, including CNN architecture experiments, hyperparameter tuning, and transfer learning for digit recognition.

# CNN-Based Handwritten Character Classification and Transfer Learning

## Overview

This project explores **image classification using Convolutional Neural Networks (CNNs)** with the EMNIST dataset.

The main objective was to build CNN models capable of recognizing handwritten characters, compare different network architectures and training parameters, and investigate whether a model trained on handwritten letters could be reused for handwritten digit classification using **transfer learning**.

The project was developed as part of the **Image Analysis and Object Recognition** course.

---

## Project Objectives

The project focuses on four main areas:

1. **Initial CNN Classification**

   * Train a simple CNN to classify handwritten letters from the EMNIST Letters dataset.

2. **CNN Architecture and Hyperparameter Experiments**

   * Design a deeper CNN architecture.
   * Compare different numbers of convolutional filters.
   * Experiment with different convolution kernel sizes.
   * Investigate the effect of learning rate, batch size, and number of epochs.

3. **Transfer Learning**

   * Start with a CNN trained for handwritten letter recognition.
   * Reuse its learned visual features for handwritten digit classification.
   * Replace the final 26-class classification layer with a 10-class layer.
   * Freeze the previously learned layers and train only the new classification layer.

4. **Evaluation on Self-Generated Images**

   * Test the trained models on handwritten/generated images outside the original EMNIST distribution.
   * Analyze the difference between performance on the benchmark dataset and real-world/self-generated images.

---

## Dataset

The project uses the **EMNIST dataset** provided through `torchvision`.

Two subsets are used:

* **EMNIST Letters** – 26 handwritten alphabet classes (`a`–`z`)
* **EMNIST Digits** – 10 handwritten digit classes (`0`–`9`)

The dataset is automatically downloaded by `torchvision` when the training script is executed.

The images are grayscale and have a resolution of **28 × 28 pixels**.

---

## Model Architecture

### Simple CNN

The initial model consists of:

```text
Input: 1 × 28 × 28

Conv2D
1 → 16 channels
5 × 5 kernel

Max Pooling
2 × 2

Conv2D
16 → 32 channels
5 × 5 kernel

Max Pooling
2 × 2

Flatten

Fully Connected
32 × 7 × 7 → 26 classes
```

ReLU activation is applied after the convolutional layers.

This model was used as the baseline architecture for handwritten letter classification.

---

## Complex CNN

A deeper CNN was then developed:

```text
Input: 1 × 28 × 28

Conv2D
1 → 32 channels
5 × 5 kernel

Max Pooling

Conv2D
32 → 64 channels
5 × 5 kernel

Max Pooling

Conv2D
64 → 128 channels
3 × 3 kernel

Fully Connected
6272 → 512

Fully Connected
512 → 26
```

The deeper architecture allows the network to learn increasingly complex visual features from the handwritten characters.

---

## Architecture Experiments

Two CNN variations were evaluated.

### Variation 1 – Increased Output Channels

The number of convolutional filters was increased:

```text
Conv1: 1 → 64
Conv2: 64 → 128
```

This allows the network to learn a larger number of feature maps.

**Test accuracy: approximately 93.36%**

### Variation 2 – Different Kernel Sizes

The convolutional kernel sizes were changed:

```text
Conv1: 3 × 3
Conv2: 7 × 7
```

Smaller kernels focus on local features, while larger kernels provide a larger receptive field.

**Test accuracy: approximately 92.74%**

---

## Hyperparameter Experiments

Several combinations of learning rate, batch size, and number of epochs were tested.

| Experiment | Learning Rate | Epochs | Batch Size | Test Accuracy |
| ---------- | ------------: | -----: | ---------: | ------------: |
| 1          |         0.001 |     10 |         64 |        93.88% |
| 2          |        0.0005 |     15 |        128 |        93.91% |
| 3          |         0.002 |      8 |        256 |        93.99% |

The third configuration achieved the highest recorded accuracy in these experiments:

```text
Learning Rate: 0.002
Epochs: 8
Batch Size: 256
Optimizer: Adam
Accuracy: 93.99%
```

This configuration was selected as the best-performing setup from the documented experiments.

---

## Transfer Learning

One of the main parts of the project was applying **transfer learning**.

Instead of training a digit classifier completely from scratch, the CNN trained on handwritten letters was reused.

The process was:

```text
EMNIST Letters
       ↓
Train CNN
       ↓
Learn visual features
(edges, curves, shapes, strokes)
       ↓
Replace final classification layer
26 classes → 10 classes
       ↓
Freeze earlier CNN layers
       ↓
Train final layer using EMNIST Digits
       ↓
Digit Classifier
```

The convolutional layers already contained useful low-level visual representations such as edges, curves, and shapes.

Only the final classification layer was trained for the new digit-classification task.

### Transfer Learning Result

The transfer-learning model achieved approximately:

**99.25% test accuracy**

This demonstrated that features learned from handwritten letters could be reused effectively for handwritten digit recognition.

---

## Self-Generated Image Evaluation

The models were also tested on images created outside the EMNIST dataset.

Examples are included in:

```text
self_generated_images/
├── digits/
└── letters/
```

The results were considerably worse than the EMNIST test accuracy.

This demonstrates an important machine-learning problem known as **domain shift**.

The EMNIST images have a specific style, preprocessing pipeline, image size, stroke distribution, and handwriting characteristics. Self-generated images can differ significantly in these properties.

Therefore, a model can achieve high benchmark accuracy while still performing poorly on real-world images.

---

## Key Results

| Model / Experiment             | Accuracy |
| ------------------------------ | -------: |
| Simple CNN                     |  ~93.05% |
| Complex CNN                    |  ~93.66% |
| CNN Variation 1                |  ~93.36% |
| CNN Variation 2                |  ~92.74% |
| Best Hyperparameter Experiment |  ~93.99% |
| Transfer Learning – Digits     |  ~99.25% |

The results show that:

* Increasing model complexity can improve classification performance.
* CNN architecture choices influence accuracy.
* Learning rate, batch size, and epochs affect both training time and accuracy.
* Transfer learning can significantly improve efficiency when moving to a related classification task.
* High benchmark accuracy does not necessarily guarantee good real-world generalization.

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Pillow
* EMNIST
* Convolutional Neural Networks
* Transfer Learning
* CUDA / GPU training

---

## Project Structure

```text
.
├── image_final_project.py
├── saved_models/
│   ├── best_letter_model_acc93.36_epochs8_batch256_Adam_lr0.002.pth
│   └── model_digit_acc99.20_epochs5_batch256_Adam_lr0.001.pth
├── self_generated_images/
│   ├── digits/
│   └── letters/
├── Image Analysis and Object Recognition - Final Project Report.pdf
└── README.md
```

> Note: The exact model filenames may differ depending on the training run.

---

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install torch torchvision matplotlib numpy pillow
```

---

## Running the Project

Run the main Python script:

```bash
python image_final_project.py
```

The script will:

1. Download the EMNIST dataset.
2. Load the handwritten letter dataset.
3. Train CNN architectures.
4. Evaluate the models.
5. Perform architecture and hyperparameter experiments.
6. Save trained models.
7. Perform transfer learning for digit classification.
8. Evaluate test images.
9. Test self-generated images.

### GPU

The code automatically detects whether CUDA is available:

```python
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
```

If a CUDA-compatible GPU is available, training will use the GPU. Otherwise, it will fall back to the CPU.

---

## Limitations

The main limitation observed in the project was the poor generalization of the models to self-generated images.

Possible improvements include:

* More diverse training data
* Stronger data augmentation
* Better image preprocessing
* Training with real-world handwritten samples
* Fine-tuning using real-world images
* More advanced CNN architectures
* Better normalization and input standardization

---

## Learning Outcomes

Through this project, we explored:

* Image classification
* CNN architecture design
* Convolution and pooling
* Feature maps
* ReLU activation
* Cross-entropy loss
* Adam optimization
* Hyperparameter tuning
* Model evaluation
* GPU-based deep learning
* Transfer learning
* Domain shift and generalization

---

## Authors

**Group 15**

* Jaiganasheelan Balgopal Sakar
* Shwethashree Kambalipura Venkatesh
* Praful Govardhan Telgad
* Rakesh Narasimha Murthy

---

## Course

**Image Analysis and Object Recognition**
Summer Semester 2025
