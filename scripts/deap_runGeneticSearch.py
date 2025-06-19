# scripts/deap_runGeneticSearch.py
import numpy as np
import torch.nn as nn
from deap import base, creator, tools, algorithms
from tqdm import trange
from scripts.deap_makeEvalFuntion import make_eval_function


# =========================
# 4) CONFIGURACIÓN Y EJECUCIÓN DE GA
# =========================
def run_genetic_search(train_ds, 
                       val_ds, n_classes,
                       pop_size=10, 
                       ngen=5, 
                       cxpb=0.5, 
                       mutpb=0.2,
                       sensor_cols="sensores"):
    """
    Corre un GA para optimizar (lr, batch_size, hidden_dim)
    y devuelve el mejor individuo.
    """
    # 1) Definir tipo de fitness y de individuo
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # 2) Inicializar Toolbox
    toolbox = base.Toolbox()
    # registros para cada hiperparámetro
    toolbox.register("attr_lr", np.random.uniform, 1e-4, 1e-2)
    toolbox.register("attr_bs", np.random.randint, 16, 65)
    toolbox.register("attr_hd", np.random.randint, 32, 129)
    # registro de individuo (combina los 3 atributos)
    toolbox.register("individual",
                     tools.initCycle,
                     creator.Individual,
                     (toolbox.attr_lr,
                      toolbox.attr_bs,
                      toolbox.attr_hd),
                     n=1)
    # registro de población (repite el individuo pop_size veces)
    toolbox.register("population",
                     tools.initRepeat,
                     list,
                     toolbox.individual)

    # 3) registrar evaluación, cruce, mutación y selección
    toolbox.register("evaluate",
                     make_eval_function(train_ds, 
                                        val_ds, 
                                        n_classes,
                                        sensor_cols=sensor_cols))
    toolbox.register("mate",   tools.cxBlend,   alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian,
                     mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament,
                     tournsize=3)

    # 4) Crear población inicial
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # 5) Bucle de generaciones con tqdm
    for gen in trange(ngen, desc="GA Generations"):
        # Selección
        offspring = toolbox.select(pop, len(pop))
        # Cruce y mutación
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        # Evaluar fitness de los nuevos
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        # Actualizar población y hall-of-fame
        pop[:] = offspring
        hof.update(pop)
        # Estadísticas
        record = stats.compile(pop)
        print(f"Gen {gen+1}/{ngen} — avg: {record['avg']:.4f}, max: {record['max']:.4f}")

    # 6) Mejor individuo
    best = hof[0]
    print("\n--- Mejor Individuo ---")
    print(f"lr = {best[0]:.6f}")
    print(f"batch_size = {int(best[1])}")
    print(f"hidden_dim = {int(best[2])}")
    print(f"Fitness = {best.fitness.values[0]:.4f}\n")
    return best