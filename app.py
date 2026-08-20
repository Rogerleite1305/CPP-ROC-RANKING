import base64
import datetime
import io
import math
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
import requests
import streamlit as st

# Importa o motor matemático analítico do CPP-ROC-RANKING (Eq. 1 a 18)
from cpp_engine import processar_cpp_roc_ranking

# ==========================================
# 0. CLASSE NUMBERED CANVAS (PAGINAÇÃO DINÂMICA)
# ==========================================


class NumberedCanvas(canvas.Canvas):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#475569"))

        # Rodapé com data/hora e numeração
        data_hora = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        texto_rodape = f"Relatório Gerado em: {data_hora} | Página {self._pageNumber} de {page_count}"

        self.drawRightString(A4[0] - 36, 20, texto_rodape)
        self.drawString(
            36,
            20,
            "DSS CPP-ROC RANKING — Desenvolvido por RISE/UFAL © Todos os direitos reservados",
        )

        # Linha divisória do rodapé
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 32, A4[0] - 36, 32)

        self.restoreState()


# ==========================================
# 1. FUNÇÕES AUXILIARES
# ==========================================


def redimensionar_imagem(image_file, largura, altura):
    img = Image.open(image_file)
    resample_filter = getattr(Image, "Resampling", Image).LANCZOS
    img_res = img.resize((largura, altura), resample_filter)
    return img_res


def imagem_para_base64(image_file, largura=44, altura=44):
    try:
        if hasattr(image_file, "getvalue"):
            image_file = io.BytesIO(image_file.getvalue())
        img_res = redimensionar_imagem(image_file, largura, altura)
        buf = io.BytesIO()
        img_res.save(buf, format="PNG")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def formatar_cnpj(cnpj_raw):
    if not cnpj_raw:
        return ""
    limpo = re.sub(r"[^a-zA-Z0-9]", "", str(cnpj_raw)).upper()
    if len(limpo) == 14:
        return (
            f"{limpo[:2]}.{limpo[2:5]}.{limpo[5:8]}/{limpo[8:12]}-{limpo[12:]}"
        )
    return limpo


def callback_atualizar_cnpj():
    cnpj_digitado = st.session_state.get("emp_cnpj_input", "")
    st.session_state["emp_cnpj"] = formatar_cnpj(cnpj_digitado)


@st.cache_data(ttl=3600, show_spinner=False)
def buscar_endereco_hibrido(cep_raw):
    if not cep_raw:
        return "", ""
    cep_limpo = re.sub(r"[^a-zA-Z0-9]", "", str(cep_raw)).upper()
    if len(cep_limpo) < 3:
        return cep_limpo, ""
    if len(cep_limpo) == 8 and cep_limpo.isdigit():
        try:
            res = requests.get(
                f"https://viacep.com.br/ws/{cep_limpo}/json/", timeout=4
            )
            if res.status_code == 200:
                dados = res.json()
                if "erro" not in dados:
                    partes = [
                        p
                        for p in [
                            dados.get("logradouro", ""),
                            dados.get("bairro", ""),
                            f"{dados.get('localidade', '')} - {dados.get('uf', '')}",
                            "Brasil",
                        ]
                        if p
                    ]
                    return cep_limpo, ", ".join(partes)
        except Exception:
            pass
    return cep_limpo, ""


def callback_atualizar_cep():
    cep_digitado = st.session_state.get("input_cep", "")
    cep_limpo, end_encontrado = buscar_endereco_hibrido(cep_digitado)
    st.session_state["cep_formatado"] = cep_limpo
    if end_encontrado:
        st.session_state["emp_end"] = end_encontrado


def gerar_grafico_ranking(df_ranking, para_impressao=False):
    figsize = (6.5, 3.2) if not para_impressao else (6.5, 2.5)
    dpi = 300 if para_impressao else 150

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    y_pos = np.arange(len(df_ranking))

    # Inverter ordem para mostrar o 1º lugar no topo
    df_sorted = df_ranking.iloc[::-1]

    ax.barh(
        y_pos,
        df_sorted["Score CPP-RANKING"],
        align="center",
        color="#2563eb",
        height=0.55,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["Alternativa"], fontsize=8)
    ax.set_xlabel("Score Global Agregado (Eq. 16–18)", fontsize=9)
    ax.set_title("Ranking das Alternativas", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ==========================================
# 2. GERADOR DE RELATÓRIO PDF A4 COM REPORTLAB
# ==========================================


def gerar_pdf_relatorio(
    df_ranking,
    resultado_completo,
    n_dms,
    n_alt,
    n_crit,
    nomes_criterios,
    tipos_criterios,
    nomes_dms,
    empresa_nome,
    empresa_cnpj,
    empresa_contato,
    empresa_cep,
    empresa_end,
    empresa_link,
    logo_file,
):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=45,
    )
    elements = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8,
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=10,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=6,
        spaceAfter=4,
    )
    analise_style = ParagraphStyle(
        "AnaliseText",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#1e293b"),
        leading=11,
    )
    footer_style = ParagraphStyle(
        "FooterText",
        parent=styles["Normal"],
        fontSize=7.5,
        textColor=colors.HexColor("#475569"),
        leading=10,
    )

    # CABEÇALHO DO DOCUMENTO
    elements.append(
        Paragraph("RELATÓRIO EXECUTIVO DE TOMADA DE DECISÃO", title_style)
    )
    elements.append(
        Paragraph(
            "SISTEMA CPP-ROC RANKING | ORDENAÇÃO COMPLETA MULTICRITÉRIO (EQUAÇÕES 1 A 18)",
            subtitle_style,
        )
    )
    elements.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#2563eb"),
            spaceAfter=8,
        )
    )

    # 1. TABELA DE RANKING FINAL
    vencedor_nome = df_ranking.iloc[0]["Alternativa"]
    vencedor_score = df_ranking.iloc[0]["Score CPP-RANKING"]

    rec_data = [["Posição", "Alternativa", "Score CPP-RANKING"]]
    for _, row in df_ranking.iterrows():
        rec_data.append(
            [f"{int(row['Posição'])}º", str(row["Alternativa"]), f"{row['Score CPP-RANKING']:.4f}"]
        )

    t_rec = Table(rec_data, colWidths=[80, 260, 140])
    t_rec.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(t_rec)
    elements.append(Spacer(1, 6))

    # 2. ANÁLISE GRÁFICA
    buf_grafico = gerar_grafico_ranking(df_ranking, para_impressao=True)
    img_grafico = RLImage(buf_grafico, width=460, height=160)

    texto_analitico = f"""
    <b>Interpretação Executiva para a Problemática de Ordenação (Ranking):</b><br/>
    • A alternativa classificada em 1º lugar foi <b>{vencedor_nome}</b> com Score Global de <b>{vencedor_score:.4f}</b>.<br/>
    • A ordenação considera a integração da matriz de probabilidade de dominância par a par $P_{{ij}}$ (Eq. 8–12) com a ponderação ROC (Eq. 1) entre os {n_dms} decisores.
    """
    p_analitico = Paragraph(texto_analitico, analise_style)

    quadro_conteudo = [
        [Paragraph("<b>Análise Gráfica e Diretriz Decisória</b>", section_style)],
        [img_grafico],
        [Spacer(1, 2)],
        [p_analitico],
    ]
    t_quadro = Table(quadro_conteudo, colWidths=[480])
    t_quadro.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    elements.append(KeepTogether([t_quadro]))
    elements.append(Spacer(1, 8))

    # 3. RASTREABILIDADE: ROC E CRITÉRIOS
    elements.append(
        Paragraph(
            "Rastreabilidade: Pesos ROC por Decisor e Tipos de Critérios",
            section_style,
        )
    )
    roc_headers = (
        ["Critério", "Tipo"]
        + [f"{dm}" for dm in nomes_dms]
        + ["Peso Médio ROC"]
    )
    roc_rows = [roc_headers]

    detalhes = resultado_completo["detalhes_dms"]
    for c_idx, c_nome in enumerate(nomes_criterios):
        c_tipo = tipos_criterios[c_idx]
        row = [str(c_nome), str(c_tipo)]
        pesos_crit = []
        for d in range(n_dms):
            p_val = detalhes[d]["pesos_criterios"][c_idx]
            pesos_crit.append(p_val)
            row.append(f"{p_val:.4f}")
        row.append(f"{np.mean(pesos_crit):.4f}")
        roc_rows.append(row)

    col_w_base = 320 // (n_dms + 1)
    t_roc = Table(
        roc_rows, colWidths=[110, 50] + [col_w_base] * (n_dms + 1)
    )
    t_roc.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f8f9fa")],
                ),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    elements.append(KeepTogether([t_roc]))

    elements.append(Spacer(1, 10))
    elements.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=colors.HexColor("#cbd5e1"),
            spaceAfter=6,
        )
    )

    # EMBEDDING DE IDENTIDADE CORPORATIVA E DIREITOS AUTORAIS NO RODAPÉ DO PDF
    logo_cell = ""
    if logo_file:
        try:
            logo_bytes = io.BytesIO(logo_file.getvalue())
            mini_logo = redimensionar_imagem(logo_bytes, 40, 40)
            mini_buffer = io.BytesIO()
            mini_logo.save(mini_buffer, format="PNG")
            mini_buffer.seek(0)
            logo_cell = RLImage(mini_buffer, width=40, height=40)
        except Exception:
            logo_cell = ""

    url_rise = "https://www.instagram.com/rise.ufal/"

    info_text = f"<b>Organização Solicitante:</b> {empresa_nome if empresa_nome else 'Não informada'}"
    if empresa_link:
        url_val = empresa_link if empresa_link.startswith("http") else f"https://{empresa_link}"
        info_text += f" | <a href='{url_val}' color='#2563eb'><u>Site Oficial</u></a>"

    info_text += f"<br/><b>CNPJ:</b> {empresa_cnpj if empresa_cnpj else 'N/A'} | <b>Contato:</b> {empresa_contato if empresa_contato else 'N/A'}"
    info_text += f" | <b>CEP:</b> {empresa_cep if empresa_cep else 'N/A'}<br/>"
    info_text += f"<b>Desenvolvimento & Direitos Autorais:</b> RISE — Laboratório de Pesquisa (UFAL) | <a href='{url_rise}' color='#2563eb'><u>@rise.ufal</u></a>"

    text_p = Paragraph(info_text, footer_style)

    if logo_cell != "":
        footer_table_data = [[logo_cell, text_p]]
        col_widths = [45, 435]
    else:
        footer_table_data = [[text_p]]
        col_widths = [480]

    t_footer = Table(footer_table_data, colWidths=col_widths)
    t_footer.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    elements.append(KeepTogether([t_footer]))

    doc.build(elements, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()


# ==========================================
# 3. INTERFACE STREAMLIT
# ==========================================

st.set_page_config(
    page_title="DSS | CPP-ROC RANKING", page_icon="📈", layout="wide"
)

# ESTILIZAÇÃO CSS
st.markdown(
    """
    <style>
    .stApp { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .main-header { background-color: #0f172a; padding: 20px; border-radius: 8px; color: #ffffff; margin-bottom: 20px; }
    .main-header h1 { color: #ffffff !important; font-size: 24px; font-weight: 600; margin: 0; }
    .main-header p { color: #94a3b8; font-size: 13px; margin-top: 4px; margin-bottom: 0; }
    .stButton>button { background-color: #2563eb !important; color: white !important; font-weight: 600 !important; border-radius: 6px !important; border: none !important; width: 100%; }
    
    .recommendation-card {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    .footer-dark-card {
        background-color: #0f172a;
        color: #ffffff;
        padding: 16px 20px;
        border-radius: 8px;
        text-align: center;
        font-size: 13px;
        margin-top: 40px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .footer-dark-card p {
        margin: 0;
        color: #e2e8f0;
    }
    .footer-dark-card a {
        color: #60a5fa !important;
        text-decoration: underline;
        font-weight: 600;
    }
    .footer-dark-card a:hover {
        color: #93c5fd !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

if "cep_formatado" not in st.session_state:
    st.session_state["cep_formatado"] = ""
if "emp_end" not in st.session_state:
    st.session_state["emp_end"] = ""
if "emp_cnpj" not in st.session_state:
    st.session_state["emp_cnpj"] = ""
if "calculo_executado" not in st.session_state:
    st.session_state["calculo_executado"] = False

# BARRA LATERAL
st.sidebar.markdown(
    """
    <div style="padding-bottom: 10px;">
        <h3 style="margin: 0; color: #0f172a; font-size: 18px;">⚙️ Configurações</h3>
        <p style="margin: 0; color: #64748b; font-size: 11px;">Parâmetros do Modelo e Simulação</p>
    </div>
""",
    unsafe_allow_html=True,
)

with st.sidebar.expander("Estrutura & Escala do Problema", expanded=True):
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        n_dms = st.number_input(
            "Nº Decisores",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            key="cfg_n_dms",
        )
        n_crit = st.number_input(
            "Nº Critérios",
            min_value=1,
            max_value=10,
            value=3,
            key="cfg_n_crit",
        )
    with col_d2:
        n_alt = st.number_input(
            "Nº Alternativas",
            min_value=2,
            max_value=20,
            value=3,
            key="cfg_n_alt",
        )

    st.markdown("---")
    tipo_escala = st.selectbox(
        "Escala dos Dados Brutos:",
        [
            "[0, 1] - Normalizada",
            "[0, 10] - Notas de 0 a 10",
            "[0, 100] - Porcentagem/Pontos",
        ],
        index=0,
        key="cfg_tipo_escala",
    )

    if "[0, 10]" in tipo_escala:
        val_max = 10.0
    elif "[0, 100]" in tipo_escala:
        val_max = 100.0
    else:
        val_max = 1.0

with st.sidebar.expander("Motor de Incerteza & Ponderação", expanded=False):
    min_std = st.number_input(
        "Variância Mínima (Estabilidade Numérica)",
        min_value=1e-6,
        max_value=1e-2,
        value=1e-4,
        format="%.6f",
        key="cfg_min_std",
    )

with st.sidebar.expander("Rotulagem e Tipos de Critério", expanded=False):
    st.markdown("**Alternativas**")
    nomes_alt_globais = [
        st.text_input(
            f"Alt {a+1}",
            value=st.session_state.get(f"glob_alt_{a}", f"Alternativa {a+1}"),
            key=f"glob_alt_{a}",
        )
        for a in range(n_alt)
    ]

    st.markdown("---")
    st.markdown("**Critérios & Direção**")
    nomes_crit_globais = []
    tipos_crit_globais = []

    for c in range(n_crit):
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            nome_c = st.text_input(
                f"Crit {c+1}",
                value=st.session_state.get(
                    f"glob_crit_{c}", f"Critério {c+1}"
                ),
                key=f"glob_crit_{c}",
            )
            nomes_crit_globais.append(nome_c)
        with col_c2:
            tipo_c = st.selectbox(
                "Tipo",
                ["Benefício", "Custo"],
                index=0
                if st.session_state.get(f"tipo_crit_{c}", "Benefício")
                == "Benefício"
                else 1,
                key=f"tipo_crit_{c}",
                help="Benefício: Maior é melhor | Custo: Menor é melhor",
            )
            tipos_crit_globais.append(tipo_c)

with st.sidebar.expander("Identidade Corporativa (PDF)", expanded=False):
    logo_file = st.file_uploader(
        "Logomarca do Cliente", type=["png", "jpg", "jpeg"], key="logo"
    )
    st.text_input(
        "Empresa / Organização",
        placeholder="Ex: RISE / UFAL",
        key="emp_nome",
    )
    st.text_input(
        "CNPJ",
        placeholder="Apenas números ou formatado",
        key="emp_cnpj_input",
        on_change=callback_atualizar_cnpj,
    )
    st.text_input(
        "Contato / E-mail", placeholder="contato@empresa.com", key="emp_contato"
    )
    st.text_input(
        "Website / Rede Social",
        value="https://www.instagram.com/rise.ufal/",
        key="emp_link",
    )
    st.text_input(
        "CEP",
        placeholder="00000-000",
        key="input_cep",
        on_change=callback_atualizar_cep,
    )
    st.text_area("Endereço", key="emp_end", height=70)

# LOGO NO TOPO
col_l1, col_l2, col_l3 = st.columns([1, 1, 1])
with col_l2:
    try:
        url_logo_github = "https://raw.githubusercontent.com/Rogerleite1305/CPP-ROC-CHOISE/main/LOGO%20CPP-ROC-RANKING.png"
        st.image(url_logo_github, width=220)
    except Exception:
        pass

# CABEÇALHO PRINCIPAL
st.markdown(
    """
    <div class="main-header">
        <h1>Decision Support System — CPP-ROC RANKING</h1>
        <p>Sistema de Apoio à Decisão Multicritério com Múltiplos Decisores Sob Incerteza Para a Problemática de Ordenação (Ranking - Eq. 1 a 18)</p>
    </div>
""",
    unsafe_allow_html=True,
)

tab_app, tab_sens, tab_manual = st.tabs(
    ["Painel de Avaliação", "Análise de Sensibilidade", "Manual do Método"]
)

with tab_manual:
    st.markdown(
        """
    ### **MANUAL DO USUÁRIO — MÉTODO CPP-ROC RANKING (EQ. 1 A 18)**

    O **CPP-ROC RANKING** combina a **Composição Probabilística de Preferências (CPP)** com os pesos **ROC (Rank Order Centroid)** para gerar uma ordenação completa de alternativas sob múltiplos critérios e julgamentos de decisores.

    ---

    #### **1. Estrutura Matemática das 18 Equações**
    1. **Pesos ROC (Eq. 1):** Ponderação baseada no ranking do critério $w_k = \frac{1}{N} \sum_{j=k}^N \frac{1}{j}$.
    2. **Estimativa Estocástica (Eq. 1–7):** Mapeamento das notas dos decisores em média $\mu_{ij}$ e variância $\sigma_{ij}^2$.
    3. **Dominância Par a Par (Eq. 8–12):** Cálculo de $P(X_i > X_j)$ via distribuição acumulada normal com a função Erro ($\text{erf}$).
    4. **Score Global e Ranking (Eq. 13–18):** Agregação da matriz $P_{ij}$ ponderada pelos pesos ROC, computando o score e a posição das alternativas.
    """
    )

with tab_app:
    matrizes_dms = []
    ordens_dms = []
    perfis_dms = []
    nomes_dms_finais = []

    st.markdown("### **Painel de Entrada de Dados dos Decisores**")

    abas_dms = st.tabs([f"Decisor {d+1}" for d in range(n_dms)])

    for d, aba in enumerate(abas_dms):
        with aba:
            col_id1, col_id2 = st.columns([1, 2])
            with col_id1:
                nome_dm = st.text_input(
                    f"Identificação do Decisor {d+1}:",
                    value=f"Decisor {d+1}",
                    key=f"nome_dm_input_{d}",
                )
                nomes_dms_finais.append(nome_dm)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("#### **Matriz de Avaliação de Alternativas**")
                valores_iniciais = np.round(
                    np.random.rand(n_alt, n_crit) * val_max, 2
                )
                df_init = pd.DataFrame(
                    valores_iniciais,
                    columns=nomes_crit_globais,
                    index=nomes_alt_globais,
                )

                df_editada = st.data_editor(
                    df_init, key=f"matriz_indiv_{d}_v2"
                )
                matrizes_dms.append(df_editada.values)

            with col2:
                st.markdown("#### **Ordenação de Preferência (ROC)**")
                st.caption("Selecione a ordem de importância dos critérios (1º = mais importante):")
                ordem_indices = []
                criterios_disponiveis = list(range(n_crit))

                for r in range(n_crit):
                    crit_selecionado_idx = st.selectbox(
                        f"{r+1}º Lugar de Importância:",
                        options=criterios_disponiveis,
                        format_func=lambda x: nomes_crit_globais[x],
                        key=f"dm_{d}_rank_{r}",
                    )
                    ordem_indices.append(crit_selecionado_idx)

                ordens_dms.append(ordem_indices)

    st.markdown("---")
    if st.button("🚀 EXECUTAR ANÁLISE DE DECISÃO", key="btn_executar"):
        st.session_state["calculo_executado"] = True

    if st.session_state.get("calculo_executado", False):
        # Execução Agregada para Múltiplos Decisores
        matriz_3d = np.array(matrizes_dms)
        scores_agregados = np.zeros(n_alt)
        detalhes_dms = []

        for d in range(n_dms):
            res_dm = processar_cpp_roc_ranking(
                matriz_3d=matriz_3d[d:d+1, :, :],
                ordem_criterios=ordens_dms[d],
                tipos_criterios=tipos_crit_globais,
                min_std=min_std,
            )
            scores_agregados += res_dm["score"]
            detalhes_dms.append(
                {
                    "decisor": d + 1,
                    "pesos_criterios": res_dm["pesos_roc"],
                    "score_dm": res_dm["score"],
                }
            )

        score_final = scores_agregados / n_dms
        ordem_indices = np.argsort(score_final)[::-1]
        posicoes = np.empty_like(ordem_indices)
        posicoes[ordem_indices] = np.arange(1, n_alt + 1)

        res_final = {
            "score": score_final,
            "posicoes": posicoes,
            "ordem_indices": ordem_indices,
            "detalhes_dms": detalhes_dms,
        }

        st.session_state["ultimo_resultado"] = res_final

        df_res = pd.DataFrame(
            {
                "Posição": posicoes,
                "Alternativa": nomes_alt_globais,
                "Score CPP-RANKING": score_final,
            }
        ).sort_values(by="Posição")

        st.markdown("### 🏆 Resultado Final do Ranking")

        st.dataframe(
            df_res.style.format({"Score CPP-RANKING": "{:.4f}"}).highlight_min(
                subset=["Posição"], color="#d1fae5"
            ),
            use_container_width=True,
        )

        vencedor_nome = nomes_alt_globais[ordem_indices[0]]
        vencedor_score = score_final[ordem_indices[0]]

        st.markdown(
            f"""
            <div class="recommendation-card">
                <h4 style="margin:0; color:#15803d;">🥇 Alternativa Recomendada (1º Lugar): <b>{vencedor_nome}</b></h4>
                <p style="margin:4px 0 0 0; color:#166534; font-size:14px;">
                    Score Global Agregado: <b>{vencedor_score:.4f}</b> considerando a agregação de todos os {n_dms} decisores.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Botão para Download do PDF
        pdf_bytes = gerar_pdf_relatorio(
            df_ranking=df_res,
            resultado_completo=res_final,
            n_dms=n_dms,
            n_alt=n_alt,
            n_crit=n_crit,
            nomes_criterios=nomes_crit_globais,
            tipos_criterios=tipos_crit_globais,
            nomes_dms=nomes_dms_finais,
            empresa_nome=st.session_state.get("emp_nome", ""),
            empresa_cnpj=st.session_state.get("emp_cnpj", ""),
            empresa_contato=st.session_state.get("emp_contato", ""),
            empresa_cep=st.session_state.get("cep_formatado", ""),
            empresa_end=st.session_state.get("emp_end", ""),
            empresa_link=st.session_state.get("emp_link", ""),
            logo_file=logo_file,
        )

        st.download_button(
            label="📄 Baixar Relatório Executivo em PDF",
            data=pdf_bytes,
            file_name=f"Relatorio_CPP_ROC_RANKING_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )

# RODAPÉ
st.markdown(
    """
    <div class="footer-dark-card">
        <p><b>DSS CPP-ROC RANKING</b> — Desenvolvido por <a href="https://www.instagram.com/rise.ufal/" target="_blank">RISE / UFAL</a> © Todos os direitos reservados</p>
    </div>
""",
    unsafe_allow_html=True,
)
