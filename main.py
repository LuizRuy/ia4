import cv2
import numpy as np
import random
import os
import argparse
from neural_network import MaskOptimizationNN
from mask_analyzer import MaskAnalyzer
from genetic_trainer import GeneticTrainer


def check_density(image, x, y, radius=10, threshold=128):
    """
    !!! NOVO !!!
    Verifica se existe uma 'massa' de preto real ou se é só uma sujeira/borda.
    Retorna True se for um objeto válido, False se for ruído.
    """
    h, w = image.shape
    x, y = int(x), int(y)

    # Recorte local
    y1, y2 = max(0, y - radius), min(h, y + radius)
    x1, x2 = max(0, x - radius), min(w, x + radius)
    roi = image[y1:y2, x1:x2]

    if roi.size == 0: return False

    # Conta quantos pixels escuros existem nessa vizinhança
    dark_pixels = np.count_nonzero(roi < threshold)

    # Se tiver menos de 20 pixels pretos num raio de 10px, é sujeira/borda fina.
    # Uma bola real teria centenas de pixels.
    if dark_pixels < 20:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Arquivo da imagem')
    parser.add_argument('--threshold', type=int, default=128, help='Limiar para busca global')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None: return print("Erro ao abrir imagem")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    analyzer = MaskAnalyzer(gray)
    trainer = GeneticTrainer(analyzer, pop_size=80)

    weights_file = "brain_v2.npy"
    best_weights = None
    if os.path.exists(weights_file):
        print(">>> Memória carregada do disco!")
        best_weights = np.load(weights_file)

    circles_found = 0
    max_fails = 5000
    fails = 0

    print(f"--- INICIANDO CAÇADA (Threshold: {args.threshold}) ---")

    while fails < max_fails:
        # A. Busca Global
        cx = random.randint(20, w - 20)
        cy = random.randint(20, h - 20)

        # 1. Checa se o pixel central é claro
        if gray[cy, cx] > args.threshold:
            fails += 1
            continue

        # !!! NOVO: Checa se é 'sujeira' ou resto de borda !!!
        if not check_density(gray, cx, cy, radius=10, threshold=args.threshold):
            # É preto, mas é muito pequeno/fino. Ignora.
            # (Isso impede o loop nas bordas que sobrarem)
            fails += 1
            continue

        fails = 0

        current_mask = (cx, cy, random.randint(15, 30))
        print(f"\n>>> ALVO POSSÍVEL EM {current_mask[:2]}")

        patience = 5
        locked_on = False

        for i in range(patience):
            new_mask, weights, fitness = trainer.train_step(
                current_mask,
                generations=15,
                previous_weights=best_weights
            )

            # Visualização
            vis = img.copy()
            # Desenha o alvo atual (Verde)
            cv2.circle(vis, (int(new_mask[0]), int(new_mask[1])), int(new_mask[2]), (0, 255, 0), 2)
            cv2.putText(vis, f"Fit: {fitness:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Hunter AI", vis)
            cv2.waitKey(10)

            if i == 0 and fitness < 0:
                print(f"   X Descartado rápido (Fitness {fitness:.1f})")
                break

            # Critério de Sucesso
            if fitness > 2200:  # Ajuste conforme necessidade
                print(f"   >>> SUCESSO CONFIRMADO! Fit: {fitness:.1f}")

                best_weights = weights
                np.save(weights_file, best_weights)

                # !!! CORREÇÃO CRÍTICA: APAGADOR AGRESSIVO !!!
                # Pinta de BRANCO na imagem de busca (gray)
                # Adiciona +4 pixels no raio para garantir que apague a borda e a sombra
                eraser_radius = int(new_mask[2]) + 4

                cv2.circle(gray, (int(new_mask[0]), int(new_mask[1])), eraser_radius, 255, -1)

                # Na imagem visual (colorida), desenha só a borda para você ver
                cv2.circle(img, (int(new_mask[0]), int(new_mask[1])), int(new_mask[2]), (0, 255, 0), 2)
                cv2.putText(img, str(circles_found + 1), (int(new_mask[0]), int(new_mask[1])), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

                circles_found += 1
                locked_on = True
                break

            current_mask = new_mask

        if not locked_on:
            print("   - Perdeu o rastro.")

    print(f"\n=== FIM === Total Círculos: {circles_found}")
    cv2.imshow("Final", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()