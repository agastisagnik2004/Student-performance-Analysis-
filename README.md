
# 🎓 Student Performance Prediction using PyTorch

This project demonstrates how deep learning techniques can be applied to predict student outcomes using **real-time educational data** collected from **New Integrated Govt. School, Ramnagar-II**. Using PyTorch, we build and evaluate a custom neural network model that takes in key academic features to predict student performance.

---

## 📊 Dataset

* **Source**: Collected from *New Integrated Govt. School, Ramnagar-II*
* **Features Used**:

  * `Date of Admission`
  * `% of Marks Scored at Previous School`
* **Target**:

  * `Roll Number` (used as a proxy for predicting academic placement/order)

---

## 🧠 Model Overview

A custom feedforward neural network (multi-layer perceptron) was implemented using **PyTorch**:

```python
self.fc1 = nn.Linear(2, 16)
self.fc2 = nn.Linear(16, 8)
self.fc3 = nn.Linear(8, 2)
self.out = nn.Linear(2, 1)
```

* **Activation**: ReLU
* **Loss Function**: Mean Squared Error (MSELoss)
* **Optimizer**: Stochastic Gradient Descent (SGD)
* **Evaluation Metric**: Root Mean Squared Error (RMSE)

---

## ⚙️ Training Pipeline

1. Data preprocessing (date formatting, dropping irrelevant columns, feature normalization)
2. Splitting data into training and test sets
3. Custom PyTorch dataset and dataloader creation
4. Model training using mini-batches
5. Evaluation on unseen test data

---

## 📈 Results

* **Test Loss**: Reported as average MSE per batch
* **RMSE**: Interpreted to show average prediction error
* **Approx. % Error**: Provided for interpretability when target is percentage-based

---

## 🚀 Future Work

* Add more features (e.g., gender, attendance, subject-wise scores)
* Use classification models for performance bands (e.g., high/medium/low performers)
---

## 🏫 Acknowledgment

> Data used in this project was collected from **New Integrated Govt. School, Ramnagar-II**.
> Special thanks to the school for providing access to real-time academic records for research and educational purposes.

---

## 💻 Requirements

* Python 3.8+
* PyTorch



