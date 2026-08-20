# CPP-ROC-RANKING

Aplicação em Python/Streamlit desenvolvida para suporte à tomada de decisão multicritério baseada na **Composição Probabilística de Preferências (CPP)** integrada aos pesos **ROC (Rank Order Centroid)** para a ordenação completa de alternativas (*Ranking*) sob o julgamento de múltiplos decisores.

---

## 📌 Sobre o Método

O **CPP-ROC-RANKING** é uma extensão probabilística que permite agregar julgamentos incertos de múltiplos decisores e critérios de diferentes naturezas (Benefício e Custo). 

A abordagem matemática desdobra-se nas **18 Equações fundamentais**:
1. **Determinação dos Pesos ROC (Eq. 1):** Atribuição de pesos surrogados aos critérios com base na ordem de preferência.
2. **Modelagem Probabilística (Eq. 1–7):** Mapeamento das avaliações em distribuições de probabilidade normais, calculando médias ($\mu$) e variâncias ($\sigma^2$) agregadas por alternativa e critério.
3. **Matriz de Dominância Par a Par (Eq. 8–12):** Avaliação analítica da probabilidade de uma alternativa superar outra ($P_{ij}$) em cada critério utilizando a função erro ($\text{erf}$).
4. **Agregação e Score Global (Eq. 13–18):** Consolidação da matriz de dominância global ponderada pelos pesos ROC e geração do *Score* final para a ordenação completa das alternativas.

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.10+**
* **Streamlit:** Interface web interativa.
* **NumPy:** Computação vetorial e matrizes multidimensionais.
* **Pandas:** Manipulação e exibição tabular de dados.
* **SciPy:** Funções matemáticas de distribuição acumulada.

---

## 📁 Estrutura do Repositório

```text
├── app.py              # Interface gráfica e fluxo de interação (Streamlit)
├── cpp_engine.py       # Motor matemático (Eq. 1 a 18 da CPP-ROC-RANKING)
├── requirements.txt    # Dependências do projeto
└── README.md           # Documentação do repositório