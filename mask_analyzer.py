import cv2
import numpy as np


class MaskAnalyzer:
    def __init__(self, image_gray):
        self.image = image_gray
        self.h, self.w = image_gray.shape

    def calculate_contrast_fitness(self, x, y, radius, edge_margin=5):
        """
        Fitness Inteligente: Diferença entre a média do anel externo e do círculo interno.
        Cria um gradiente suave para a IA subir.
        """
        if radius < 3: return -1000.0

        ix, iy, ir = int(x), int(y), int(radius)

        # Otimização: Recorte (ROI)
        pad = ir + edge_margin + 2
        y1, y2 = max(0, iy - pad), min(self.h, iy + pad)
        x1, x2 = max(0, ix - pad), min(self.w, ix + pad)
        roi = self.image[y1:y2, x1:x2]
        if roi.size == 0: return -1000.0

        lx, ly = ix - x1, iy - y1

        # Máscaras
        m_inner = np.zeros(roi.shape, np.uint8)
        cv2.circle(m_inner, (lx, ly), ir, 255, -1)

        m_outer = np.zeros(roi.shape, np.uint8)
        cv2.circle(m_outer, (lx, ly), ir + edge_margin, 255, -1)
        m_ring = cv2.subtract(m_outer, m_inner)

        # Médias de Intensidade
        mean_in = cv2.mean(roi, mask=m_inner)[0]
        mean_out = cv2.mean(roi, mask=m_ring)[0]

        # Fitness Base: Contraste (Fora Claro - Dentro Escuro)
        score = (mean_out - mean_in) * 10

        # Penalidades para guiar a IA
        # 1. Se dentro for muito claro (>100), não é bola
        if mean_in > 100: score -= (mean_in - 100) * 5

        # 2. Se fora for muito escuro (<150), o círculo está grande demais
        if mean_out < 150: score -= (150 - mean_out) * 5

        return score

    def get_sensors(self, x, y, radius):
        """
        14 Sensores Relativos (A IA não sabe X/Y absoluto).
        Ela aprende: 'Está mais escuro à direita? Mova para a direita'.
        """
        sensors = []
        # Ângulos: N, NE, E, SE, S, SW, W, NW
        angles = np.linspace(-np.pi, np.pi, 9)[:-1]

        # ROI para velocidade
        pad = int(radius + 5)
        y1, y2 = max(0, int(y) - pad), min(self.h, int(y) + pad)
        x1, x2 = max(0, int(x) - pad), min(self.w, int(x) + pad)
        roi = self.image[y1:y2, x1:x2]
        lx, ly = int(x) - x1, int(y) - y1

        if roi.size == 0: return np.zeros(14)

        # 8 Sensores de Borda (Detectam se o círculo está cortando a bola)
        for ang in angles:
            # Ponto de teste na borda do círculo atual
            sx = int(lx + radius * np.cos(ang))
            sy = int(ly + radius * np.sin(ang))

            val = 255
            if 0 <= sx < roi.shape[1] and 0 <= sy < roi.shape[0]:
                val = roi[sy, sx]
            sensors.append(val / 255.0)  # Normaliza 0-1

        # 4 Sensores Diferenciais (Direção do gradiente)
        # N-S, E-W, NE-SW, NW-SE
        sensors.append(sensors[0] - sensors[4])  # Vertical
        sensors.append(sensors[2] - sensors[6])  # Horizontal
        sensors.append(sensors[1] - sensors[5])  # Diagonal 1
        sensors.append(sensors[7] - sensors[3])  # Diagonal 2

        # 1 Sensor de Preenchimento Médio
        m_inner = np.zeros(roi.shape, np.uint8)
        cv2.circle(m_inner, (lx, ly), int(radius), 255, -1)
        mean_val = cv2.mean(roi, mask=m_inner)[0]
        sensors.append(mean_val / 255.0)

        # 1 Sensor de Proporção (Tamanho relativo à imagem - Opcional, mas ajuda)
        sensors.append(radius / min(self.h, self.w))

        return np.array(sensors)  # Total 14