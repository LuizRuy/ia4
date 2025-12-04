import numpy as np


class MaskOptimizationNN:
    """
    Rede Neural 'Piloto': Recebe o que os sensores veem e decide para onde mover o círculo.
    Entrada: 14 sensores (visão local).
    Saída: 3 comandos (Mover X, Mover Y, Mudar Raio).
    """

    def __init__(self, input_size=14, hidden_size=16, output_size=3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicialização Xavier/Glorot
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, hidden_size))

        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        if x.ndim == 1: x = x.reshape(1, -1)

        # Camadas Ocultas (ReLU)
        self.a1 = self.relu(np.dot(x, self.W1) + self.b1)
        self.a2 = self.relu(np.dot(self.a1, self.W2) + self.b2)

        # Camada de Saída (Tanh) -> Saída entre -1 e 1
        # Isso é perfeito para "Mover Esquerda/Direita" ou "Aumentar/Diminuir"
        self.output = np.tanh(np.dot(self.a2, self.W3) + self.b3)
        return self.output

    def predict_movement(self, features, max_move=20, max_resize=10):
        """Traduz a saída da rede (-1 a 1) em pixels."""
        out = self.forward(features)[0]

        # out[0] -> Raio, out[1] -> X, out[2] -> Y
        d_radius = out[0] * max_resize
        d_x = out[1] * max_move
        d_y = out[2] * max_move

        return int(d_x), int(d_y), int(d_radius)

    def get_weights(self):
        return np.concatenate([w.flatten() for w in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]])

    def set_weights(self, flat_weights):
        idx = 0
        sizes = [
            (self.input_size, self.hidden_size), (1, self.hidden_size),
            (self.hidden_size, self.hidden_size), (1, self.hidden_size),
            (self.hidden_size, self.output_size), (1, self.output_size)
        ]
        params = []
        for shape in sizes:
            size = np.prod(shape)
            params.append(flat_weights[idx:idx + size].reshape(shape))
            idx += size

        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = params

    def save(self, path):
        np.save(path, self.get_weights())

    def load(self, path):
        self.set_weights(np.load(path))