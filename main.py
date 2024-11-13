import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc

# Cargar el conjunto de datos de entrenamiento desde un archivo CSV
train_data = pd.read_csv("train.csv")

# Convertir la columna de 'Sex' a valores numéricos (1 para masculino, 0 para femenino)
train_data['Sex'] = train_data['Sex'].map({'male': 1, 'female': 0})

# Extraer el título de cada pasajero a partir del nombre
train_data['Name title'] = train_data['Name'].str.extract(r',\s*([^\.]*)\.')  # Captura el título antes del punto

# Mapear los títulos a valores numéricos definidos según la especificación
title_mapping = {
    'Mrs': 1, 'Miss': 2, 'Dr': 3, 'Rev': 3, 'Don': 4, 'Jonkheer': 4,
    'Capt': 5, 'Col': 5, 'Master': 6, 'Mr': 7
}
# Los títulos no mapeados se asignan al valor 8
train_data['Name title'] = train_data['Name title'].map(title_mapping).fillna(8)

# Crear variables dummy para la columna 'Embarked' (puerto de embarque)
train_data = pd.get_dummies(train_data, columns=['Embarked'], prefix='Embarked', drop_first=True)

# Rellenar los valores faltantes en 'Age' y 'Fare' con la mediana de cada columna
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())

# Eliminar columnas que no aportan información útil para el modelo
train_data = train_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Separar las características predictoras (X) de la variable objetivo (y)
X = train_data.drop(columns=['Survived'])
y = train_data['Survived']

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión logística y entrenarlo con los datos de entrenamiento
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
predictions = model.predict(X_test)

# Verificar que las dimensiones de las predicciones coinciden con las de las etiquetas reales
print(f"Dimensiones de y_test: {y_test.shape}")
print(f"Dimensiones de predictions: {predictions.shape}")

# Si las dimensiones no coinciden, imprimir una advertencia
if y_test.shape[0] != predictions.shape[0]:
    print(f"Advertencia: El tamaño de las muestras no coincide entre y_test y predictions.")
else:
    # Generar y mostrar la matriz de confusión
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No sobrevivió", "Sobrevivió"])
    disp.plot(cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.show()

# Graficar la distribución de las predicciones
plt.figure(figsize=(6, 4))
plt.hist(predictions, bins=2, rwidth=0.8, color='green', edgecolor='black', align='mid')
plt.xticks([0, 1], ['No sobrevivió', 'Sobrevivió'])
plt.xlabel('Predicción')
plt.ylabel('Número de personas')
plt.title('Distribución de Predicciones (Sobrevivió vs No sobrevivió)')
plt.show()

# Crear un gráfico de pastel para mostrar la distribución de supervivencia
plt.figure(figsize=(6, 6))
survived_count = np.sum(predictions == 1)
not_survived_count = np.sum(predictions == 0)
plt.pie([not_survived_count, survived_count], labels=['No sobrevivió', 'Sobrevivió'], autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
plt.title('Distribución de Predicciones (Sobrevivió vs No sobrevivió)')
plt.show()

# Mostrar nuevamente la matriz de confusión para evaluar el desempeño del modelo
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No sobrevivió", "Sobrevivió"])
disp.plot(cmap='Blues')
plt.title('Matriz de Confusión')
plt.show()

# Imprimir un reporte detallado de la clasificación, incluyendo precisión, recall, f1-score
report = classification_report(y_test, predictions, target_names=["No sobrevivió", "Sobrevivió"])
print("Reporte de Clasificación:")
print(report)

# Cargar y preprocesar el conjunto de datos de prueba de la misma manera que el conjunto de entrenamiento
test_data = pd.read_csv("test.csv")
test_data['Sex'] = test_data['Sex'].map({'male': 1, 'female': 0})
test_data['Name title'] = test_data['Name'].str.extract(r',\s*([^\.]*)\.')
test_data['Name title'] = test_data['Name title'].map(title_mapping).fillna(8)
test_data = pd.get_dummies(test_data, columns=['Embarked'], prefix='Embarked', drop_first=True)
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
test_data = test_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Asegurarse de que las columnas del conjunto de prueba coincidan con las del conjunto de entrenamiento
X_test_final = test_data.reindex(columns=X.columns, fill_value=0)

# Calcular la curva ROC y el área bajo la curva (AUC)
y_prob = model.predict_proba(X_test)[:, 1]  # Obtener las probabilidades de la clase positiva
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Calcular las tasas de falsos positivos y verdaderos positivos
roc_auc = auc(fpr, tpr)  # Calcular el área bajo la curva

# Graficar la curva ROC
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Realizar validación cruzada y mostrar las puntuaciones obtenidas
scores = cross_val_score(model, X, y, cv=5)
print(f"Scores de validación cruzada: {scores}")
print(f"Precisión media: {scores.mean():.2f}")
