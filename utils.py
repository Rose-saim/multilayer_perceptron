
# --- Entraînement et prédiction ---
def forward(network, X):
    activations = []
    input = X
    for l in network:
        activations.append(l.forward(input))
        input = activations[-1]
    return activations

def predict_(network, X):
    logits = forward(network, X)[-1]
    return logits.argmax(axis=-1)
