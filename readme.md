# Supervised Model Comparison: Perceptron vs. Decision Tree

This project presents a comparative study between a simple linear neural network model (**Perceptron**) and a **Decision Tree** classifier applied to the *Wine* dataset.

## Wine Dataset
<img src="./wine.png" width="500px" height="500px" alt="ERRO">
<center>
<a href="https://archive.ics.uci.edu/dataset/109/wine">
https://archive.ics.uci.edu/dataset/109/wine</a>
</center>

## 📊 Dataset
* **Dataset:** Wine Dataset (UCI / Scikit-Learn)
* **Samples:** 178
* **Features:** 13 chemical analyses
* **Classes:** 3 types of wine (Class 0, Class 1, Class 2)

## 🛠️ Preprocessing and Modeling

| Model | Preprocessing | Configuration / Hyperparameters |
| :--- | :--- | :--- |
| **Perceptron** | Min-Max Normalization | `eta0=0.1`, `max_iter=1000`, `random_state=42` |
| **Decision Tree** | Univariate Selection (`SelectKBest`, k=5, `f_classif`) | Default `DecisionTreeClassifier` settings |

## 📈 Results

| Model | Accuracy | Notes |
| :--- | :--- | :--- |
| **Perceptron** | **100.0%** | Converged in 14 iterations |
| **Decision Tree** | **97.2%** | Using the top 5 selected features |

## 🧠 Theoretical Considerations

* **Perceptron:** A linear algorithm based on a single artificial neuron. Achieved perfect performance after proper feature scaling.
* **Decision Tree:** A hierarchical model based on decision rules (Gini impurity). Maintained high accuracy even after reducing dimensionality to just 5 features.

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Korzre/wine-classifier-perceptron-vs-tree.git