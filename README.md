# CNN-and-ViT-from-scratch
this project aims to  compare a vision transformer with a CNN for image classifaction tasks. Both the CNN and ViT are built from scratch using pytorch and then trained on the CIFAR-10 dataset



### Objective:
To empirically evaluate and compare the performance of a custom CNN and ViT, focusing on their learning efficiency (convergence speed), parameter usage, and ability to handle spatial relationships on a small-scale dataset without pre-training.
## Dataset
This project uses the CIFAR-10 dataset. It contains 60,000 32×32 color images (50,000 for training and 10,000 for testing) across 10 different classes:

- Transport: Airplane, Automobile, Ship, Truck.

- Animals: Bird, Cat, Deer, Dog, Frog, Horse.

## Methodology
The comparison is built on two distinct architectural philosophies:

- CNN (Convolutional Neural Network): Implemented using hierarchical layers that utilize local receptive fields (kernels) and pooling to extract spatial features.

- ViT (Vision Transformer): Implemented by partitioning images into 4×4 patches, projecting them into embeddings, and using a Multi-Head Self-Attention (MHSA) mechanism to capture global dependencies.


### Training setup:

Hardware: Trained on GPU using CUDA for accelerated computation.

trained each model for 20 epochs only

Data Pipeline: Used the standard PyTorch DataLoader with pin_memory=True and num_workers to handle asynchronous batch-by-batch transfers to VRAM.

Optimizer: AdamW with a learning rate of 1×10−3 and weight decay for regularization.

Loss Function: CrossEntropyLoss.

Preprocessing: Normalization to a mean/std of 0.5 and resizing to 32×32 pixels.

## Results

The performance was tracked via training loss and accuracy history:

### Loss graph:

![Loss Curves](loss_curves.png)

CNN Performance: Showed rapid initial convergence. The built-in inductive bias allowed the model to reach respectable accuracy within the first 5 epochs.

ViT Performance: Demonstrated a slower "warm-up" phase. Because the model must learn spatial relationships from scratch via positional embeddings, the loss curve was initially flatter compared to the CNN.

### Acuaracy graph

![accuaracy curves](accuracy_curves.png)

Final Accuracy: [CNN: 92%, ViT: 80%]

## Discussion

The results highlight the fundamental trade-offs between the two architectures:

Inductive Bias vs. Global Attention: The CNN's assumption that pixels are locally related makes it highly efficient for small datasets like CIFAR-10. The ViT, while more "flexible" globally, is highly data-hungry and typically requires larger datasets or heavy augmentation to outperform a CNN.

Sensitivity: The ViT showed higher sensitivity to hyperparameters (like learning rate and weight decay) compared to the CNN, which was more robust during the training process.

Conclusion: For small-scale image classification tasks with limited data, a scratch-built CNN remains the more efficient and performant choice over a scratch-built Transformer.


