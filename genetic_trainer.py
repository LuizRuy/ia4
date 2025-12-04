import numpy as np
from deap import base, creator, tools
import random
from neural_network import MaskOptimizationNN

# Setup DEAP Global
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)


class GeneticTrainer:
    def __init__(self, analyzer, pop_size=50, mutation_rate=0.2):
        self.analyzer = analyzer
        self.pop_size = pop_size
        self.mut_rate = mutation_rate
        self.nn_ref = MaskOptimizationNN()  # Referência para estrutura
        self.num_weights = len(self.nn_ref.get_weights())

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, -1, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_float, n=self.num_weights)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def train_step(self, start_mask, generations=10, previous_weights=None):
        """Roda uma evolução curta para melhorar a máscara atual."""
        pop = self.toolbox.population(n=self.pop_size)

        # --- INJEÇÃO DE MEMÓRIA ---
        if previous_weights is not None:
            # 1. O Mestre (Cópia exata)
            pop[0][:] = previous_weights
            # 2. Os Aprendizes (Mutações do mestre)
            for i in range(1, int(self.pop_size * 0.4)):  # 40% da população
                ind = creator.Individual(previous_weights)
                self.toolbox.mutate(ind)
                pop[i][:] = ind

        # Função de Avaliação Interna
        def evaluate(ind):
            self.nn_ref.set_weights(np.array(ind))

            # IA propõe movimento
            feats = self.analyzer.get_sensors(*start_mask)
            dx, dy, dr = self.nn_ref.predict_movement(feats)

            nx = start_mask[0] + dx
            ny = start_mask[1] + dy
            nr = max(5, start_mask[2] + dr)

            # Limites da imagem
            h, w = self.analyzer.h, self.analyzer.w
            nx = max(0, min(w, nx))
            ny = max(0, min(h, ny))

            fit = self.analyzer.calculate_contrast_fitness(nx, ny, nr)
            return (fit,)

        self.toolbox.register("evaluate", evaluate)

        # Loop Evolutivo Rápido
        for g in range(generations):
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.6:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mut_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Avalia
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring

        best_ind = tools.selBest(pop, 1)[0]

        # Retorna a máscara resultante e os pesos vencedores
        self.nn_ref.set_weights(np.array(best_ind))
        feats = self.analyzer.get_sensors(*start_mask)
        dx, dy, dr = self.nn_ref.predict_movement(feats)

        final_mask = (
            int(start_mask[0] + dx),
            int(start_mask[1] + dy),
            int(max(5, start_mask[2] + dr))
        )

        return final_mask, np.array(best_ind), best_ind.fitness.values[0]