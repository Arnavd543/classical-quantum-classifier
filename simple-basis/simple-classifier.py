import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(weights, x):
    qml.AngleEmbedding(x, wires=[0, 1])  # encodes classical features into qubit states
    qml.BasicEntanglerLayers(weights, wires=[0, 1])  # adds trainable layers
    return qml.expval(qml.PauliZ(0))  # measure qubit 0 in Z-basis

def quantum_model(weights, x):
    return circuit(weights, x)

np.random.seed(42)
weights = np.random.randn(3, 2)

opt = qml.AdamOptimizer(stepsize=0.01)
epochs = 50

for epoch in range(epochs):
    for x_i, y_i in zip(X_train, y_train):
        loss = (quantum_model(weights, x_i) - y_i) ** 2
        weights = opt.step(lambda w: (quantum_model(w, x_i) - y_i) ** 2, weights)

predictions = [np.sign(quantum_model(weights, x)) for x in X_test]
acc = np.mean(predictions == (2 * y_test - 1))  # map y in {0,1} → {-1,+1}
print(f"Test accuracy: {acc:.2f}")

