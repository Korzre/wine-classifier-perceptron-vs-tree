#Usando Perceptron
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import MinMaxScaler

wine = load_wine()
X = wine.data
y = wine.target


# Dividindo o dataset em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Normalização min-max
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Treinando o perceptron
perceptron = Perceptron(eta0=0.1, max_iter=1000, random_state=42)
perceptron.fit(X_train, y_train)

numero_iter = perceptron.n_iter_
print(f"Número de iterações necessárias: {numero_iter}")

# Calculando a acurácia
accuracy = perceptron.score(X_test, y_test) * 100
print(f"Acurácia: {round(accuracy,1)}%")

#Usando árvores de decisão

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

wine = load_wine()
X = wine.data
y = wine.target

# Dividindo o dataset em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicando a seleção de atributos univariada
# Seleciona os 5 melhores atributos (k=5)
selector = SelectKBest(score_func=f_classif, k=5)

X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Treinando o classificador com os atributos selecionados
clf = DecisionTreeClassifier()
clf.fit(X_train_selected, y_train)

# Fazendo as previsões no conjunto de teste
y_pred = clf.predict(X_test_selected)

# Calculando a acurácia
accuracy = accuracy_score(y_test, y_pred)
accuracy_percent = accuracy * 100
print(f"Acurácia: {round(accuracy_percent, 1)}%")