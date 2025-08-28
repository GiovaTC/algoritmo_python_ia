# algoritmo_python_ia

<img width="1283" height="1079" alt="image" src="https://github.com/user-attachments/assets/8937cbdc-42d8-4f6a-9df6-25ac4ba3bb14" />

# 🌸 Clasificador de Flores Iris con Scikit-learn

Este programa entrena un modelo de inteligencia artificial para predecir si una flor es del tipo **Setosa**, **Versicolor** o **Virginica**, utilizando el famoso dataset **Iris**.

---

## 🧠 Librería a utilizar

- [scikit-learn](https://scikit-learn.org/): Librería de machine learning en Python.

### 📝 Requisitos

Antes de ejecutar el código, asegúrate de tener instalada la librería:

```bash
pip install scikit-learn
💡 Código completo en Python
python
Copiar código
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Cargar el dataset Iris
iris = load_iris()
X = iris.data  # características (largo y ancho de sépalos y pétalos)
y = iris.target  # etiquetas (0: setosa, 1: versicolor, 2: virginica)

# 2. Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Crear el modelo K-Nearest Neighbors (KNN)
modelo = KNeighborsClassifier(n_neighbors=3)

# 4. Entrenar el modelo
modelo.fit(X_train, y_train)

# 5. Realizar predicciones
y_pred = modelo.predict(X_test)

# 6. Calcular precisión
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {precision * 100:.2f}%")

# 7. Probar con un nuevo dato
nueva_flor = [[5.1, 3.5, 1.4, 0.2]]  # ejemplo con características de una flor
prediccion = modelo.predict(nueva_flor)
print(f"La flor es de tipo: {iris.target_names[prediccion[0]]}")

📌 ¿Qué hace este programa?
✅ Carga un dataset real de flores Iris.
✅ Entrena un modelo de clasificación usando K-Nearest Neighbors (KNN).
✅ Evalúa la precisión del modelo con datos no vistos.
✅ Predice el tipo de flor para una nueva muestra basada en sus características.

✅ Resultado esperado (ejemplo)

Precisión del modelo: 100.00%
La flor es de tipo: setosa
⚠️ Nota: La precisión puede variar levemente dependiendo de la división aleatoria del dataset.

📚 Recursos adicionales
Documentación oficial de scikit-learn

Información sobre el dataset Iris en Wikipedia
