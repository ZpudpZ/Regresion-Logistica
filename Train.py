import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
# 1. Cargar dataset de entrenamiento
train_data = pd.read_csv("train.csv")

# 2. Transformar columnas
# Sexo a numérico
train_data['Sex'] = train_data['Sex'].map({'male': 1, 'female': 0})

# Extraer títulos del nombre
train_data['Name title'] = train_data['Name'].str.extract(r',\s*([^\.]*)\.')  # Extrae el título del nombre

# títulos a los códigos proporcionados
title_mapping = {
    'Mrs': 1, 'Miss': 2, 'Dr': 3, 'Rev': 3, 'Don': 4, 'Jonkheer': 4,
    'Capt': 5, 'Col': 5, 'Master': 6, 'Mr': 7
}
train_data['Name title'] = train_data['Name title'].map(title_mapping).fillna(8)

# dummies para "Embarked"
train_data = pd.get_dummies(train_data, columns=['Embarked'], prefix='Embarked', drop_first=True)

# 3. Rellenar valores faltantes
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())

# 4. Eliminar columnas
train_data = train_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Exportar el dataset transformado a un archivo CSV
train_data.to_csv("train_processed.csv", index=False)
print("El archivo procesado 'train_processed.csv' ha sido generado.")

# 5. Separar variables
X = train_data.drop(columns=['Survived'])
y = train_data['Survived']

# 6. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Entrenar el modelo de regresión logística
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)

print(f"Dimensiones de y_train: {y_train.shape}")
print(f"Dimensiones de train_predictions: {train_predictions.shape}")

if y_train.shape[0] != train_predictions.shape[0]:
    print(f"Advertencia: El tamaño de las muestras no coincide entre y_train y train_predictions.")
else:
    # 8. Evaluación de los resultados
    cm = confusion_matrix(y_train, train_predictions)
    print("Matriz de Confusión en Entrenamiento:")
    print(cm)

    # Reporte de Clasificación
    report = classification_report(y_train, train_predictions, target_names=["No sobrevivió", "Sobrevivió"])
    print("Reporte de Clasificación (Entrenamiento):")
    print(report)

    # Curva ROC y AUC para el conjunto de entrenamiento
    y_prob_train = model.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, y_prob_train)
    roc_auc = auc(fpr, tpr)
    print(f"Área bajo la curva ROC : {roc_auc:.2f}")

# 9. Validación Cruzada (Para ver la efectividad de la generalización)
scores = cross_val_score(model, X, y, cv=5)
print(f"Scores de validación cruzada: {scores}")
print(f"Precisión media: {scores.mean():.2f}")
