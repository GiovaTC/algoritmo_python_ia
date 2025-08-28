from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Cargar el dataset Iris
iris = load_iris()
X = iris.data # caracteristicas (largo y ancho de sepalos y petalos)
y = iris.target # etiquetas (0: setosa, 1: versicolor, 2: virginica)

# 2. Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Crear el modelo K-Nearest Neighbors (KNN)
modelo = KNeighborsClassifier(n_neighbors=3)

# 4. Entrenar el modelo
modelo.fit(X_train, y_train)

# 5. Realizar predicciones
y_pred = modelo.predict(X_test)

# 6. Calcular precision
precision = accuracy_score(y_test, y_pred)
print(f"Precision del modelo: {precision *  100:.2f}%")

# 7. Probar con nuevos datos
ejemplos = [
    [5.1, 3.5, 1.4, 0.2], # setosa
    [6.0, 2.7, 4.5, 1.5], # versicolor
    [6.7, 3.0, 5.2, 2.3], # virginica
    [7.2, 3.6, 6.1, 2.5], # virginica
    [5.9, 3.0, 4.2, 1.5], # versicolor
]

print("\npredicciones para nuevas flores:")
for i, flor in enumerate(ejemplos, 1):
    prediccion = modelo.predict([flor])
    tipo = iris.target_names[prediccion[0]]
    print(f"ejemplo {i}: {flor} -> {tipo}")
