import numpy as np
import pandas as pd
import streamlit as st
from cpp_engine import processar_cpp_roc_ranking

st.set_page_config(
    page_title="CPP-ROC-RANKING", page_icon="📊", layout="wide"
)

st.title("📊 CPP-ROC-RANKING")
st.markdown(
    "Aplicação para ordenação de alternativas sob múltiplos critérios e decisores (Eq. 1 a 18)."
)

# --- CONFIGURAÇÕES LATERAIS ---
st.sidebar.header("⚙️ Configurações do Problema")
n_decisores = st.sidebar.number_input(
    "Nº de Decisores", min_value=1, value=2, step=1
)
n_alternativas = st.sidebar.number_input(
    "Nº de Alternativas", min_value=2, value=4, step=1
)
n_criterios = st.sidebar.number_input(
    "Nº de Critérios", min_value=1, value=3, step=1
)

st.sidebar.subheader("🎯 Tipos de Critérios")
tipos_criterios = []
for c in range(n_criterios):
    tipo = st.sidebar.selectbox(
        f"Critério {c+1}",
        options=["Benefício", "Custo"],
        key=f"tipo_crit_{c}",
    )
    tipos_criterios.append(tipo)

# --- NOMEAÇÃO DAS ALTERNATIVAS E CRITÉRIOS ---
st.header("📝 Nomes e Estrutura dos Dados")

cols_alt = st.columns(min(n_alternativas, 4))
nomes_alternativas = []
for i in range(n_alternativas):
    col_idx = i % 4
    nome = cols_alt[col_idx].text_input(
        f"Alt {i+1}", value=f"Alternativa {i+1}", key=f"alt_name_{i}"
    )
    nomes_alternativas.append(nome)

nomes_criterios = [f"Critério {c+1}" for c in range(n_criterios)]

# --- ENTRADA DE NOTAS DOS DECISORES ---
st.subheader("📋 Avaliações dos Decisores")
matrizes_decisores = []

tabs = st.tabs([f"👤 Decisor {d+1}" for d in range(n_decisores)])
for d, tab in enumerate(tabs):
    with tab:
        df_init = pd.DataFrame(
            np.random.randint(1, 10, size=(n_alternativas, n_criterios)),
            index=nomes_alternativas,
            columns=nomes_criterios,
        )
        df_edited = st.data_editor(
            df_init, key=f"editor_decisor_{d}", use_container_width=True
        )
        matrizes_decisores.append(df_edited.values)

matriz_3d = np.array(matrizes_decisores)

# --- ORDENAÇÃO DE IMPORTÂNCIA DOS CRITÉRIOS ---
st.header("⚖️ Preferência dos Critérios (ROC)")
ordem_criterios = st.multiselect(
    "Selecione os critérios na ordem do MAIS importante para o LEVES importante:",
    options=list(range(n_criterios)),
    format_func=lambda x: f"Critério {x+1} ({tipos_criterios[x]})",
    default=list(range(n_criterios)),
)

# --- EXECUÇÃO DO MODELO ---
st.markdown("---")
if len(ordem_criterios) != n_criterios:
    st.warning("⚠️ Selecione e ordene todos os critérios para habilitar a execução.")
else:
    if st.button("🚀 Processar Ranking Final", type="primary"):
        res = processar_cpp_roc_ranking(
            matriz_3d=matriz_3d,
            ordem_criterios=ordem_criterios,
            tipos_criterios=tipos_criterios,
        )

        st.header("🏆 Resultado Final do Ranking")

        df_ranking = pd.DataFrame(
            {
                "Posição": res["posicoes"],
                "Alternativa": nomes_alternativas,
                "Score CPP-RANKING": res["score"],
            }
        ).sort_values(by="Posição")

        st.dataframe(
            df_ranking.style.format({"Score CPP-RANKING": "{:.4f}"}).highlight_min(
                subset=["Posição"], color="#d1fae5"
            ),
            use_container_width=True,
        )

        # Detalhes Adicionais
        with st.expander("🔍 Ver Detalhes Numéricos (Pesos e Matriz de Dominância Global)"):
            st.subheader("Pesos ROC Calculados")
            df_pesos = pd.DataFrame(
                {
                    "Critério": nomes_criterios,
                    "Tipo": tipos_criterios,
                    "Peso ROC": res["pesos_roc"],
                }
            )
            st.dataframe(
                df_pesos.style.format({"Peso ROC": "{:.4f}"}),
                use_container_width=True,
            )

            st.subheader("Matriz de Dominância Global Pij")
            df_pij = pd.DataFrame(
                res["matriz_dominancia_global"],
                index=nomes_alternativas,
                columns=nomes_alternativas,
            )
            st.dataframe(
                df_pij.style.format("{:.4f}"), use_container_width=True
            )