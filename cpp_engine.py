import math
import numpy as np


def calcular_pesos_roc(n_crit):
    """Calcula os pesos ROC (Rank Order Centroid) para n critérios."""
    pesos = np.zeros(n_crit)
    for i in range(n_crit):
        pesos[i] = np.sum([1 / j for j in range(i + 1, n_crit + 1)]) / n_crit
    return pesos


def processar_cpp_roc_ranking(
    matriz_3d, ordem_criterios, tipos_criterios, min_std=1e-4
):
    """
    Executa o modelo analítico CPP-ROC-RANKING (Eq. 1 a 18).

    matriz_3d: array numpy (n_decisores, n_alternativas, n_criterios)
    ordem_criterios: lista com a ordenação dos índices dos critérios
    tipos_criterios: lista ('Benefício' ou 'Custo') para cada critério
    """
    matriz = np.array(matriz_3d, dtype=float)
    n_decisores, n_alt, n_crit = matriz.shape

    # 1. Normalização / Inversão para critérios de Custo
    for c in range(n_crit):
        if tipos_criterios[c] == "Custo":
            matriz[:, :, c] = -matriz[:, :, c]

    # 2. Pesos ROC Ponderados pela Ordem de Preferência (Eq. 1)
    pesos_roc_base = calcular_pesos_roc(n_crit)
    pesos_roc = np.zeros(n_crit)
    for rank, crit_idx in enumerate(ordem_criterios):
        pesos_roc[crit_idx] = pesos_roc_base[rank]

    # 3. Média e Variância por Alternativa/Critério (Eq. 1 a 7)
    medias = np.mean(matriz, axis=0)
    variancias = (
        np.var(matriz, axis=0, ddof=1) if n_decisores > 1 else np.zeros((n_alt, n_crit))
    )
    variancias = np.nan_to_num(variancias, nan=0.0)

    # 4. Matriz de Dominância Par a Par P_ij via Distribuição Normal acumulada (Eq. 8 a 12)
    prob_dominancia = np.zeros((n_alt, n_alt, n_crit))
    for c in range(n_crit):
        for i in range(n_alt):
            for j in range(n_alt):
                if i != j:
                    diff_media = medias[i, c] - medias[j, c]
                    std_comb = np.sqrt(
                        max(variancias[i, c] + variancias[j, c], min_std)
                    )
                    z = diff_media / (std_comb * np.sqrt(2))
                    prob_dominancia[i, j, c] = 0.5 * (1 + math.erf(z))
                else:
                    prob_dominancia[i, j, c] = 0.5

    # 5. Agregação Global Ponderada pelos Pesos ROC (Eq. 13 a 15)
    matriz_dominancia_global = np.zeros((n_alt, n_alt))
    for i in range(n_alt):
        for j in range(n_alt):
            matriz_dominancia_global[i, j] = np.sum(
                prob_dominancia[i, j, :] * pesos_roc
            )

    # 6. Score Final e Ordenação do Ranking (Eq. 16 a 18)
    score = np.sum(matriz_dominancia_global, axis=1)
    ordem_indices = np.argsort(score)[::-1]
    posicoes = np.empty_like(ordem_indices)
    posicoes[ordem_indices] = np.arange(1, n_alt + 1)

    return {
        "score": score,
        "posicoes": posicoes,
        "ordem_indices": ordem_indices,
        "pesos_roc": pesos_roc,
        "matriz_dominancia_global": matriz_dominancia_global,
        "medias": medias,
        "variancias": variancias,
    }