# Thermal Profile Prediction using Physics-Informed Neural Networks (PINN)

## Overview

This project applies **Physics-Informed Neural Networks (PINN)** to estimate the full thermal profile of a toy problem. The approach combines deep learning with physical constraints to predict temperature distributions across spatial domains.

## Problem Description

The goal is to predict the complete temperature field given:
- **Input**: 
  - Spatial coordinates (x, y, z positions)
  - Values from thermal couples (sensor measurements)
- **Output**: 
  - Temperature at any point in the spatial domain

This is a typical inverse problem in thermal engineering where we want to reconstruct the full temperature field from sparse sensor measurements.

## Physics-Informed Neural Networks (PINN)

Physics-Informed Neural Networks integrate physical laws (e.g., heat equation, boundary conditions) directly into the neural network training process. This approach offers several advantages:
- Reduces the amount of data required for training
- Ensures predictions satisfy physical constraints
- Improves generalization to unseen conditions
- Provides physically consistent solutions

## Implementation

The machine learning model is built using **PyTorch**, leveraging its automatic differentiation capabilities to enforce physical constraints during training. The PINN architecture typically includes:
- Neural network to approximate the temperature field
- Loss function combining data fitting and physics-based terms
- Automatic differentiation to compute spatial derivatives required by governing equations

## Project Structure

This is a toy problem implementation designed to demonstrate the PINN methodology for thermal profile prediction. The project serves as a foundation for more complex thermal analysis applications.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- (Additional dependencies to be specified)

## Future Work

- Complete implementation of the PINN model
- Add training and inference scripts
- Include visualization tools for thermal profiles
- Provide example datasets and use cases