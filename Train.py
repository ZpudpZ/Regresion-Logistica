import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc

# 1. Cargar dataset de entrenamiento
train_data = pd.read_csv("train.csv")

# 2. Transformar columnas en el dataset de entrenamiento
# Sexo a numérico
train_data['Sex'] = train_data['Sex'].map({'male': 1, 'female': 0})

# Extraer títulos del nombre
train_data['Name title'] = train_data['Name'].str.extract(r',\s*([^\.]*)\.')  # Extrae el título del nombre

# Mapear títulos a los códigos proporcionados
title_mapping = {
    'Mrs': 1, 'Miss': 2, 'Dr': 3, 'Rev': 3, 'Don': 4, 'Jonkheer': 4,
    'Capt': 5, 'Col': 5, 'Master': 6, 'Mr': 7
}
train_data['Name title'] = train_data['Name title'].map(title_mapping).fillna(8)  # Otros títulos se mapean a 8

# Crear dummies para "Embarked"
train_data = pd.get_dummies(train_data, columns=['Embarked'], prefix='Embarked', drop_first=True)

# 3. Rellenar valores faltantes en el dataset de entrenamiento
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())  # Rellenar edad con la mediana
train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())  # Rellenar tarifa con la mediana

# 4. Eliminar columnas innecesarias en el dataset de entrenamiento
train_data = train_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])  # Estas columnas no son útiles para el modelo

# 5. Separar variables predictoras (X) y objetivo (y) en el dataset de entrenamiento
X = train_data.drop(columns=['Survived'])
y = train_data['Survived']

# 6. Dividir en entrenamiento y prueba (solo si deseas hacer validación)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Entrenar el modelo de regresión logística
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Hacer predicciones con el modelo entrenado usando el conjunto de prueba
train_predictions = model.predict(X_train)

# Verificar si las dimensiones de y_train y train_predictions coinciden
print(f"Dimensiones de y_train: {y_train.shape}")
print(f"Dimensiones de train_predictions: {train_predictions.shape}")

# Asegúrate de que las dimensiones sean las mismas
if y_train.shape[0] != train_predictions.shape[0]:
    print(f"Advertencia: El tamaño de las muestras no coincide entre y_train y train_predictions.")
else:
    # 8. Evaluación de los resultados
    cm = confusion_matrix(y_train, train_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No sobrevivió", "Sobrevivió"])
    disp.plot(cmap='Blues')
    plt.title('Matriz de Confusión en Entrenamiento')
    plt.show()

    # a. Gráfico de Distribución de Predicciones
    plt.figure(figsize=(6, 4))
    plt.hist(train_predictions, bins=2, rwidth=0.8, color='green', edgecolor='black', align='mid')
    plt.xticks([0, 1], ['No sobrevivió', 'Sobrevivió'])
    plt.xlabel('Predicción')
    plt.ylabel('Número de personas')
    plt.title('Distribución de Predicciones (Entrenamiento)')
    plt.show()

    # b. Gráfico de Pastel (Pie)
    plt.figure(figsize=(6, 6))
    survived_count = np.sum(train_predictions == 1)
    not_survived_count = np.sum(train_predictions == 0)
    plt.pie([not_survived_count, survived_count], labels=['No sobrevivió', 'Sobrevivió'], autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
    plt.title('Distribución de Predicciones (Entrenamiento)')
    plt.show()

    # c. Reporte de Clasificación
    report = classification_report(y_train, train_predictions, target_names=["No sobrevivió", "Sobrevivió"])
    print("Reporte de Clasificación (Entrenamiento):")
    print(report)

    # d. Curva ROC y AUC para el conjunto de entrenamiento
    y_prob_train = model.predict_proba(X_train)[:, 1]  # Obtener las probabilidades de clase 1 usando X_train
    fpr, tpr, thresholds = roc_curve(y_train, y_prob_train)  # Usamos y_train, que es el conjunto de entrenamiento
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Curva ROC (Entrenamiento)')
    plt.legend(loc='lower right')
    plt.show()

# 9. Validación Cruzada (Para ver la efectividad de la generalización)
scores = cross_val_score(model, X, y, cv=5)
print(f"Scores de validación cruzada: {scores}")
print(f"Precisión media: {scores.mean():.2f}")
