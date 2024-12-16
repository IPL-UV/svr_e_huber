# Support Vector Regression (SVR) and Custom Kernel Methods  
**Version:** 1.0  
**Release Date:** June 2024  

## Overview  
This repository provides implementations of Support Vector Regression (SVR) with:  
- **Huber Loss** for robust regression.  
- **Custom Kernel Functions** for advanced flexibility.  
- **Error Metrics** for model evaluation.  

It includes examples for training SVR models, computing results, and customizing kernel methods for specific regression tasks.  

## Features  
- **SVR Training**: Train SVR models using Huber loss and regularization.  
- **Custom Kernels**: Define your own kernel functions for improved performance.  
- **Performance Evaluation**: Compute accuracy metrics like MSE, RMSE, and residual errors.  
- **Precompiled Solvers**: Use optimized `.dll` files for faster execution.  

## Installation  
Clone the repository:  
```bash
git clone https://github.com/IPL-UV/svr_e_huber.git
cd svr_e_huber
```
Add paths to MATLAB:
```matlab
addpath(pwd);
```

Verify your MATLAB environment and ensure .dll files match your operating system.

---

## Usage

### **Example 1: SVR with Huber Loss**

Run `DemoEHuberSVR.m` to train an SVR model using Huber loss and Gaussian kernels.

```matlab
% Demo: SVR with Huber Loss
clear; clc; close all;

% Load data
load meris_data.mat;
X = meris_data(:, 1);  % Input features
Y = meris_data(:, 2);  % Output targets

% SVR training parameters
C = 10;        % Regularization parameter
epsilon = 0.1; % Epsilon-tube
sigma = 2;     % Gaussian kernel width

% Train SVR model
model = mysvr(X, Y, C, epsilon, sigma);

% Predict outputs
Y_pred = svroutput(X, model);

% Compute results
results = ComputeResults(Y, Y_pred);
disp('SVR Results:');
disp(results);

% Plot predictions
figure;
plot(Y, 'b'); hold on;
plot(Y_pred, 'r--');
legend('True', 'Predicted');
title('SVR Predictions with Huber Loss');
```

---

### **Example 2: Using Custom Kernels**

You can implement your own kernel function using `mysvkernel2.m`. For example:

```matlab
% Custom Gaussian kernel implementation
function K = mysvkernel2(X1, X2, sigma)
    D = pdist2(X1, X2, 'euclidean');
    K = exp(-(D.^2) / (2 * sigma^2));
end
```
To use this kernel in SVR:

```matlab
% Compute custom kernel
K = mysvkernel2(X, X, sigma);
```
---

### **Example 3: Free Parameter SVR**

Use `svrFreeParam.m` for flexible SVR training with optimized parameters:

```matlab
% Free Parameter SVR Example
C = 1; epsilon = 0.01; sigma = 1;

% Train model with free parameters
model = svrFreeParam(X, Y, C, epsilon, sigma);
```
---

## Results and Visualization

**Accuracy Metrics:**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-Squared (R²)

**Visualizations**

- Prediction vs Ground Truth plots
- Residual Analysis for error distribution

---

## Dependencies

- MATLAB R2021b or later.
- Root Mean Squared Error (RMSE)
- R-Squared (R²)

**Visualizations**

- Prediction vs Ground Truth plots
- Precompiled SVR solvers (`mysvr.dll`, `svrFreeParam.dll`).

---


## Authors


- **Gustavo Camps-Valls**  
- **Jordi Muñoz-Marí**  


**Copyright © 2006**

---

## License

This project is distributed under the **MIT License**.

---
