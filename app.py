# =============================================================================
# Dashboard Interativo — Análise de Churn de Clientes
# Executar com: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análise de Churn de Clientes",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Paleta de cores premium
# ─────────────────────────────────────────────────────────────
CORES = {
    "primaria": "#6C5CE7",
    "secundaria": "#FD79A8",
    "terciaria": "#00CEC9",
    "sucesso": "#00B894",
    "alerta": "#FDCB6E",
    "perigo": "#D63031",
    "info": "#0984E3",
    "escuro": "#2D3436",
    "claro": "#DFE6E9",
}
PALETA_CHURN = [CORES["terciaria"], CORES["secundaria"]]

# ─────────────────────────────────────────────────────────────
# CSS customizado para aparência profissional
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Fundo e texto geral */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }

    /* Cards de métricas */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 18px 20px;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(108,92,231,0.25);
    }
    div[data-testid="stMetric"] label {
        color: #a29bfe !important;
        font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #a29bfe;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #a29bfe !important;
        font-weight: 600 !important;
        border-radius: 12px 12px 0 0 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(108,92,231,0.15) !important;
        border-bottom: 3px solid #6C5CE7 !important;
        color: #ffffff !important;
    }

    /* Cabeçalho customizado */
    .main-header {
        background: linear-gradient(135deg, rgba(108,92,231,0.25), rgba(253,121,168,0.18));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 32px 36px;
        margin-bottom: 28px;
        backdrop-filter: blur(16px);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .main-header p {
        color: #b2bec3;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    /* Info boxes */
    .info-box {
        background: rgba(9,132,227,0.12);
        border-left: 4px solid #0984E3;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 12px 0;
        color: #dfe6e9;
    }
    .success-box {
        background: rgba(0,184,148,0.12);
        border-left: 4px solid #00B894;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 12px 0;
        color: #dfe6e9;
    }
    .warning-box {
        background: rgba(214,48,49,0.12);
        border-left: 4px solid #D63031;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 12px 0;
        color: #dfe6e9;
    }

    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #6C5CE7, #FD79A8, transparent);
        margin: 28px 0;
        border: none;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Carregamento e cache dos dados
# ─────────────────────────────────────────────────────────────


@st.cache_data
def carregar_dados():
    """Carrega e prepara o dataset de churn."""
    df = pd.read_csv("Churn_Modelling.csv")
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    return df


df = carregar_dados()

# ─────────────────────────────────────────────────────────────
# Sidebar — Filtros interativos
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🎛️ Filtros")
st.sidebar.markdown("---")

# Filtro de geografia
paises = st.sidebar.multiselect(
    "🌍 Geografia",
    options=sorted(df["Geography"].unique()),
    default=sorted(df["Geography"].unique()),
    help="Selecione os países para análise",
)

# Filtro de gênero
generos = st.sidebar.multiselect(
    "👤 Gênero",
    options=sorted(df["Gender"].unique()),
    default=sorted(df["Gender"].unique()),
    help="Selecione o(s) gênero(s)",
)

# Filtro de idade
st.sidebar.markdown("##### 🎂 Faixa Etária")
idade_min, idade_max = st.sidebar.slider(
    "Idade",
    min_value=int(df["Age"].min()),
    max_value=int(df["Age"].max()),
    value=(int(df["Age"].min()), int(df["Age"].max())),
    label_visibility="collapsed",
)

# Filtro de saldo
st.sidebar.markdown("##### 💰 Faixa de Saldo")
saldo_min, saldo_max = st.sidebar.slider(
    "Saldo",
    min_value=float(df["Balance"].min()),
    max_value=float(df["Balance"].max()),
    value=(float(df["Balance"].min()), float(df["Balance"].max())),
    format="%.0f",
    label_visibility="collapsed",
)

# Filtro de CreditScore
st.sidebar.markdown("##### 📈 Score de Crédito")
score_min, score_max = st.sidebar.slider(
    "CreditScore",
    min_value=int(df["CreditScore"].min()),
    max_value=int(df["CreditScore"].max()),
    value=(int(df["CreditScore"].min()), int(df["CreditScore"].max())),
    label_visibility="collapsed",
)

# Aplicar filtros
df_filtrado = df[
    (df["Geography"].isin(paises))
    & (df["Gender"].isin(generos))
    & (df["Age"].between(idade_min, idade_max))
    & (df["Balance"].between(saldo_min, saldo_max))
    & (df["CreditScore"].between(score_min, score_max))
]

st.sidebar.markdown("---")
st.sidebar.metric("Clientes filtrados", f"{len(df_filtrado):,}")
st.sidebar.metric("Total original", f"{len(df):,}")

# ─────────────────────────────────────────────────────────────
# Cabeçalho
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="main-header">
    <h1>📊 Análise de Churn de Clientes</h1>
    <p>
        Dashboard interativo para análise estatística de evasão bancária.
        Integra análise exploratória, regressão linear, testes de hipóteses
        e geração de insights acionáveis para estratégias de retenção de clientes.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Template de layout para gráficos Plotly (tema escuro)
# ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#dfe6e9", family="Inter, sans-serif"),
    title_font=dict(size=18, color="#ffffff"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#b2bec3")),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
    margin=dict(t=50, b=40, l=50, r=20),
)

# ─────────────────────────────────────────────────────────────
# Abas principais
# ─────────────────────────────────────────────────────────────
tabs = st.tabs(
    [
        "📋 Visão Geral",
        "🔍 Exploração (EDA)",
        "📐 Relações",
        "📈 Regressão Linear",
        "🔮 Previsão",
        "🧪 Testes de Hipóteses",
        "📏 Intervalos de Confiança",
        "💡 Insights & Recomendações",
    ]
)

# =================================================================
# ABA 1 — VISÃO GERAL (KPIs)
# =================================================================
with tabs[0]:
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    total_clientes = len(df_filtrado)
    taxa_churn = df_filtrado["Exited"].mean() * 100 if total_clientes > 0 else 0
    idade_media = df_filtrado["Age"].mean() if total_clientes > 0 else 0
    saldo_medio = df_filtrado["Balance"].mean() if total_clientes > 0 else 0
    score_medio = df_filtrado["CreditScore"].mean() if total_clientes > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total de Clientes", f"{total_clientes:,}")
    c2.metric("📉 Taxa de Churn", f"{taxa_churn:.1f}%")
    c3.metric("🎂 Idade Média", f"{idade_media:.1f} anos")
    c4.metric("💰 Saldo Médio", f"€ {saldo_medio:,.0f}")
    c5.metric("📊 Score Médio", f"{score_medio:.0f}")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Gráficos resumo
    col_a, col_b = st.columns(2)

    with col_a:
        if total_clientes > 0:
            contagem_churn = df_filtrado["Exited"].value_counts().reset_index()
            contagem_churn.columns = ["Status", "Quantidade"]
            contagem_churn["Status"] = contagem_churn["Status"].map(
                {0: "Permaneceu", 1: "Saiu"}
            )
            fig_pie = px.pie(
                contagem_churn,
                values="Quantidade",
                names="Status",
                title="Distribuição de Churn",
                color="Status",
                color_discrete_map={
                    "Permaneceu": CORES["terciaria"],
                    "Saiu": CORES["secundaria"],
                },
                hole=0.45,
            )
            fig_pie.update_traces(
                textinfo="percent+value",
                textfont_size=14,
                marker=dict(line=dict(color="#1a1a3e", width=2)),
            )
            fig_pie.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        if total_clientes > 0:
            churn_geo = (
                df_filtrado.groupby("Geography")["Exited"]
                .mean()
                .mul(100)
                .reset_index()
            )
            churn_geo.columns = ["País", "Taxa de Churn (%)"]
            fig_geo = px.bar(
                churn_geo,
                x="País",
                y="Taxa de Churn (%)",
                title="Taxa de Churn por País",
                color="País",
                color_discrete_sequence=[
                    CORES["primaria"],
                    CORES["secundaria"],
                    CORES["terciaria"],
                ],
                text_auto=".1f",
            )
            fig_geo.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_geo.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig_geo, use_container_width=True)

# =================================================================
# ABA 2 — EXPLORAÇÃO DOS DADOS (EDA)
# =================================================================
with tabs[1]:
    st.markdown("### 🔍 Análise Exploratória de Dados")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if total_clientes == 0:
        st.warning("Nenhum dado disponível com os filtros selecionados.")
    else:
        # Linha 1: Churn por Gênero + Churn por País
        eda_c1, eda_c2 = st.columns(2)

        with eda_c1:
            churn_gen = (
                df_filtrado.groupby("Gender")["Exited"]
                .mean()
                .mul(100)
                .reset_index()
            )
            churn_gen.columns = ["Gênero", "Taxa de Churn (%)"]
            fig_gen = px.bar(
                churn_gen,
                x="Gênero",
                y="Taxa de Churn (%)",
                title="Taxa de Churn por Gênero",
                color="Gênero",
                color_discrete_sequence=[CORES["info"], CORES["secundaria"]],
                text_auto=".1f",
            )
            fig_gen.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_gen.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig_gen, use_container_width=True)

        with eda_c2:
            churn_country = (
                df_filtrado.groupby("Geography")["Exited"]
                .mean()
                .mul(100)
                .reset_index()
            )
            churn_country.columns = ["País", "Taxa de Churn (%)"]
            fig_country = px.bar(
                churn_country,
                x="País",
                y="Taxa de Churn (%)",
                title="Taxa de Churn por País",
                color="País",
                color_discrete_sequence=[
                    CORES["primaria"],
                    CORES["terciaria"],
                    CORES["alerta"],
                ],
                text_auto=".1f",
            )
            fig_country.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_country.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig_country, use_container_width=True)

        # Linha 2: Boxplots
        eda_c3, eda_c4 = st.columns(2)

        with eda_c3:
            fig_box_age = px.box(
                df_filtrado,
                x="Exited",
                y="Age",
                color="Exited",
                title="Idade vs Churn (Boxplot)",
                color_discrete_sequence=PALETA_CHURN,
                labels={"Exited": "Churn", "Age": "Idade"},
                category_orders={"Exited": [0, 1]},
            )
            fig_box_age.update_layout(**PLOTLY_LAYOUT)
            fig_box_age.update_xaxes(
                ticktext=["Permaneceu", "Saiu"], tickvals=[0, 1]
            )
            st.plotly_chart(fig_box_age, use_container_width=True)

        with eda_c4:
            fig_box_bal = px.box(
                df_filtrado,
                x="Exited",
                y="Balance",
                color="Exited",
                title="Saldo vs Churn (Boxplot)",
                color_discrete_sequence=PALETA_CHURN,
                labels={"Exited": "Churn", "Balance": "Saldo (€)"},
                category_orders={"Exited": [0, 1]},
            )
            fig_box_bal.update_layout(**PLOTLY_LAYOUT)
            fig_box_bal.update_xaxes(
                ticktext=["Permaneceu", "Saiu"], tickvals=[0, 1]
            )
            st.plotly_chart(fig_box_bal, use_container_width=True)

        # Linha 3: Heatmap de correlação
        st.markdown("#### 🔥 Mapa de Correlação")
        cols_num = [
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Exited",
        ]
        corr = df_filtrado[cols_num].corr()

        fig_heat = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Matriz de Correlação",
            aspect="auto",
        )
        fig_heat.update_layout(
            **PLOTLY_LAYOUT,
            height=550,
            coloraxis_colorbar=dict(title="Correlação"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# =================================================================
# ABA 3 — ANÁLISE DE RELAÇÕES (Scatter interativo)
# =================================================================
with tabs[2]:
    st.markdown("### 📐 Análise de Relações entre Variáveis")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if total_clientes == 0:
        st.warning("Nenhum dado disponível com os filtros selecionados.")
    else:
        variaveis_num = [
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "EstimatedSalary",
        ]

        rel_c1, rel_c2 = st.columns(2)
        with rel_c1:
            var_x = st.selectbox("Variável X", variaveis_num, index=1, key="scatter_x")
        with rel_c2:
            var_y = st.selectbox("Variável Y", variaveis_num, index=3, key="scatter_y")

        df_scatter = df_filtrado.copy()
        df_scatter["Churn"] = df_scatter["Exited"].map({0: "Permaneceu", 1: "Saiu"})

        fig_scatter = px.scatter(
            df_scatter,
            x=var_x,
            y=var_y,
            color="Churn",
            color_discrete_map={
                "Permaneceu": CORES["terciaria"],
                "Saiu": CORES["secundaria"],
            },
            opacity=0.5,
            title=f"{var_x} vs {var_y} — colorido por Churn",
            trendline="ols",
            trendline_scope="overall",
        )
        fig_scatter.update_layout(**PLOTLY_LAYOUT, height=550)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Coeficiente de correlação
        corr_val = df_filtrado[var_x].corr(df_filtrado[var_y])
        intensidade = "forte" if abs(corr_val) > 0.5 else "moderada" if abs(corr_val) > 0.3 else "fraca"
        direcao = "positiva" if corr_val > 0 else "negativa"
        st.markdown(
            f"""
<div class="info-box">
    <strong>Correlação de Pearson:</strong> r = {corr_val:.4f} — Relação <strong>{intensidade} {direcao}</strong>
    entre {var_x} e {var_y}.
</div>
""",
            unsafe_allow_html=True,
        )

# =================================================================
# ABA 4 — MODELO DE REGRESSÃO LINEAR
# =================================================================
with tabs[3]:
    st.markdown("### 📈 Modelo de Regressão Linear")
    st.markdown(
        """
<div class="info-box">
    <strong>Nota:</strong> A variável-alvo <em>Exited</em> é binária.
    A regressão linear é aplicada aqui como Modelo Linear de Probabilidade (MLP)
    para fins didáticos. Os valores preditos são interpretados como probabilidades estimadas de churn.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Seleção de variáveis independentes
    features_disponiveis = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    features_sel = st.multiselect(
        "Selecione as variáveis independentes",
        features_disponiveis,
        default=["Age", "Balance", "IsActiveMember", "NumOfProducts"],
        key="reg_features",
    )

    if len(features_sel) < 1:
        st.warning("Selecione ao menos uma variável independente.")
    elif total_clientes < 30:
        st.warning("Dados insuficientes para ajustar o modelo com os filtros atuais.")
    else:
        # Preparar dados e one-hot encoding para Geography
        df_model = df_filtrado.copy()
        df_model = pd.get_dummies(df_model, columns=["Geography", "Gender"], drop_first=True, dtype=int)

        # Montar X com features selecionadas + dummies se existirem
        X_cols = features_sel.copy()
        for c in df_model.columns:
            if c.startswith("Geography_") or c.startswith("Gender_"):
                X_cols.append(c)

        X = df_model[X_cols].values
        y = df_model["Exited"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Métricas do modelo
        met_a, met_b = st.columns(2)
        met_a.metric("R² (Coeficiente de Determinação)", f"{r2:.4f}")
        met_b.metric("RMSE (Erro Quadrático Médio)", f"{rmse:.4f}")

        # Coeficientes
        st.markdown("#### Coeficientes do Modelo")
        coefs = pd.DataFrame(
            {"Variável": X_cols, "Coeficiente": modelo.coef_}
        ).sort_values("Coeficiente", key=abs, ascending=False)
        coefs["Impacto"] = coefs["Coeficiente"].apply(
            lambda x: "⬆️ Aumenta churn" if x > 0 else "⬇️ Reduz churn"
        )
        coefs["Magnitude"] = coefs["Coeficiente"].abs()

        fig_coef = px.bar(
            coefs,
            x="Coeficiente",
            y="Variável",
            orientation="h",
            title="Impacto das Variáveis na Probabilidade de Churn",
            color="Coeficiente",
            color_continuous_scale=["#00CEC9", "#DFE6E9", "#FD79A8"],
            color_continuous_midpoint=0,
        )
        fig_coef.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig_coef, use_container_width=True)

        # Tabela de coeficientes
        st.dataframe(
            coefs[["Variável", "Coeficiente", "Impacto"]].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        # Gráfico: Real vs Previsto
        st.markdown("#### Valores Reais vs Previstos")
        fig_pred = go.Figure()
        fig_pred.add_trace(
            go.Scatter(
                x=list(range(len(y_test))),
                y=y_test,
                mode="markers",
                name="Real",
                marker=dict(color=CORES["terciaria"], size=5, opacity=0.5),
            )
        )
        fig_pred.add_trace(
            go.Scatter(
                x=list(range(len(y_pred))),
                y=y_pred,
                mode="markers",
                name="Previsto",
                marker=dict(color=CORES["secundaria"], size=5, opacity=0.5),
            )
        )
        fig_pred.update_layout(
            **PLOTLY_LAYOUT,
            title="Comparação: Valores Reais vs Previstos",
            xaxis_title="Observação",
            yaxis_title="Valor (0 = Permaneceu, 1 = Saiu)",
            height=450,
        )
        st.plotly_chart(fig_pred, use_container_width=True)

# =================================================================
# ABA 5 — PREVISÃO INTERATIVA
# =================================================================
with tabs[4]:
    st.markdown("### 🔮 Previsão Interativa de Churn")
    st.markdown(
        """
<div class="info-box">
    Insira os dados de um cliente hipotético e veja a previsão de churn 
    gerada pelo modelo de regressão linear.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Treinar um modelo completo com o dataset inteiro (sem filtros)
    @st.cache_resource
    def treinar_modelo_completo():
        """Treina o modelo de regressão completo para previsões."""
        df_full = carregar_dados()
        df_m = pd.get_dummies(df_full, columns=["Geography", "Gender"], drop_first=True, dtype=int)
        feature_cols = [
            "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary",
        ]
        for c in df_m.columns:
            if c.startswith("Geography_") or c.startswith("Gender_"):
                feature_cols.append(c)
        X_full = df_m[feature_cols].values
        y_full = df_m["Exited"].values
        mdl = LinearRegression()
        mdl.fit(X_full, y_full)
        return mdl, feature_cols

    modelo_completo, feature_names = treinar_modelo_completo()

    prev_c1, prev_c2 = st.columns(2)

    with prev_c1:
        inp_credit = st.number_input("📊 Credit Score", min_value=300, max_value=850, value=650, step=10)
        inp_age = st.number_input("🎂 Idade", min_value=18, max_value=92, value=35, step=1)
        inp_tenure = st.number_input("📅 Tenure (anos de cliente)", min_value=0, max_value=10, value=5, step=1)

    with prev_c2:
        inp_balance = st.number_input("💰 Saldo (€)", min_value=0.0, max_value=300000.0, value=76000.0, step=1000.0)
        inp_salary = st.number_input("💵 Salário Estimado (€)", min_value=0.0, max_value=250000.0, value=100000.0, step=5000.0)
        inp_products = st.selectbox("📦 Nº de Produtos", [1, 2, 3, 4], index=0)

    prev_c3, prev_c4, prev_c5 = st.columns(3)
    with prev_c3:
        inp_card = st.selectbox("💳 Possui Cartão de Crédito?", ["Sim", "Não"], index=0)
    with prev_c4:
        inp_active = st.selectbox("✅ Membro Ativo?", ["Sim", "Não"], index=0)
    with prev_c5:
        inp_geo = st.selectbox("🌍 País", ["France", "Germany", "Spain"], index=0)
        inp_gender = st.selectbox("👤 Gênero", ["Male", "Female"], index=0)

    if st.button("🔮 Prever Churn", use_container_width=True, type="primary"):
        # Montar vetor de features na ordem correta
        input_dict = {
            "CreditScore": inp_credit,
            "Age": inp_age,
            "Tenure": inp_tenure,
            "Balance": inp_balance,
            "NumOfProducts": inp_products,
            "HasCrCard": 1 if inp_card == "Sim" else 0,
            "IsActiveMember": 1 if inp_active == "Sim" else 0,
            "EstimatedSalary": inp_salary,
        }

        # Dummies
        for feat in feature_names:
            if feat.startswith("Geography_"):
                country = feat.replace("Geography_", "")
                input_dict[feat] = 1 if inp_geo == country else 0
            elif feat.startswith("Gender_"):
                gen = feat.replace("Gender_", "")
                input_dict[feat] = 1 if inp_gender == gen else 0

        input_array = np.array([[input_dict.get(f, 0) for f in feature_names]])
        prob = modelo_completo.predict(input_array)[0]
        prob_clamped = np.clip(prob, 0, 1)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Resultado visual
        res_c1, res_c2 = st.columns([1, 2])
        with res_c1:
            st.metric("Probabilidade de Churn", f"{prob_clamped * 100:.1f}%")
            if prob_clamped > 0.5:
                st.markdown(
                    '<div class="warning-box"><strong>⚠️ ALTO RISCO de churn!</strong> Recomenda-se ação proativa de retenção.</div>',
                    unsafe_allow_html=True,
                )
            elif prob_clamped > 0.3:
                st.markdown(
                    '<div class="info-box"><strong>⚡ Risco moderado de churn.</strong> Monitorar e oferecer benefícios preventivos.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="success-box"><strong>✅ Baixo risco de churn.</strong> Cliente com boa retenção prevista.</div>',
                    unsafe_allow_html=True,
                )

        with res_c2:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob_clamped * 100,
                    number={"suffix": "%", "font": {"size": 42, "color": "#ffffff"}},
                    title={"text": "Risco de Churn", "font": {"size": 18, "color": "#a29bfe"}},
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor="#b2bec3"),
                        bar=dict(color=CORES["secundaria"] if prob_clamped > 0.5 else CORES["terciaria"]),
                        bgcolor="rgba(0,0,0,0)",
                        steps=[
                            dict(range=[0, 30], color="rgba(0,184,148,0.15)"),
                            dict(range=[30, 50], color="rgba(253,203,110,0.15)"),
                            dict(range=[50, 100], color="rgba(214,48,49,0.15)"),
                        ],
                        threshold=dict(
                            line=dict(color="#ffffff", width=3),
                            thickness=0.8,
                            value=prob_clamped * 100,
                        ),
                    ),
                )
            )
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#dfe6e9"),
                height=300,
                margin=dict(t=40, b=20, l=30, r=30),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

# =================================================================
# ABA 6 — TESTES DE HIPÓTESES
# =================================================================
with tabs[5]:
    st.markdown("### 🧪 Testes de Hipóteses")
    st.markdown(
        """
<div class="info-box">
    Utilizamos o <strong>teste t de Student para amostras independentes</strong>
    para avaliar se existe diferença significativa entre clientes que saíram e
    os que permaneceram, considerando nível de significância α = 0.05.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if total_clientes < 10:
        st.warning("Dados insuficientes para executar testes de hipótese com os filtros atuais.")
    else:
        saiu = df_filtrado[df_filtrado["Exited"] == 1]
        ficou = df_filtrado[df_filtrado["Exited"] == 0]

        def executar_teste(nome_var, label_var):
            """Executa teste t e retorna resultado formatado."""
            grupo_saiu = saiu[nome_var].dropna()
            grupo_ficou = ficou[nome_var].dropna()

            if len(grupo_saiu) < 2 or len(grupo_ficou) < 2:
                return None

            t_stat, p_valor = ttest_ind(grupo_saiu, grupo_ficou, equal_var=False)
            media_saiu = grupo_saiu.mean()
            media_ficou = grupo_ficou.mean()
            return {
                "Variável": label_var,
                "Média (Saiu)": media_saiu,
                "Média (Ficou)": media_ficou,
                "Diferença": media_saiu - media_ficou,
                "Estatística t": t_stat,
                "p-valor": p_valor,
                "Conclusão": "Rejeitamos H₀ — diferença significativa"
                if p_valor < 0.05
                else "Não rejeitamos H₀ — sem diferença significativa",
            }

        testes = [
            ("Balance", "💰 Saldo (€)"),
            ("Age", "🎂 Idade"),
            ("CreditScore", "📊 Score de Crédito"),
            ("EstimatedSalary", "💵 Salário Estimado"),
            ("Tenure", "📅 Tempo de Cliente"),
            ("NumOfProducts", "📦 Nº de Produtos"),
        ]

        resultados = []
        for var, label in testes:
            res = executar_teste(var, label)
            if res:
                resultados.append(res)

        if resultados:
            for r in resultados:
                significativo = r["p-valor"] < 0.05
                box_class = "warning-box" if significativo else "success-box"
                emoji = "❌" if significativo else "✅"

                st.markdown(
                    f"""
<div class="{box_class}">
    <strong>{r['Variável']}</strong><br>
    Média (Saiu): <strong>{r['Média (Saiu)']:,.2f}</strong> &nbsp;|&nbsp;
    Média (Ficou): <strong>{r['Média (Ficou)']:,.2f}</strong> &nbsp;|&nbsp;
    Diferença: <strong>{r['Diferença']:+,.2f}</strong><br>
    Estatística t: <strong>{r['Estatística t']:.4f}</strong> &nbsp;|&nbsp;
    p-valor: <strong>{r['p-valor']:.6f}</strong><br>
    {emoji} <em>{r['Conclusão']}</em>
</div>
""",
                    unsafe_allow_html=True,
                )

            # Tabela resumo
            st.markdown("#### 📋 Resumo dos Testes")
            df_testes = pd.DataFrame(resultados)
            st.dataframe(df_testes, use_container_width=True, hide_index=True)

# =================================================================
# ABA 7 — INTERVALOS DE CONFIANÇA
# =================================================================
with tabs[6]:
    st.markdown("### 📏 Intervalos de Confiança (95%)")
    st.markdown(
        """
<div class="info-box">
    Calculamos intervalos de confiança de 95% para as médias populacionais
    de variáveis-chave, utilizando a distribuição t de Student.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if total_clientes < 5:
        st.warning("Dados insuficientes para calcular intervalos de confiança.")
    else:

        def calc_ic(dados, confianca=0.95):
            """Calcula intervalo de confiança para a média."""
            n = len(dados)
            media = dados.mean()
            erro_padrao = stats.sem(dados)
            ic = stats.t.interval(confianca, df=n - 1, loc=media, scale=erro_padrao)
            return media, ic[0], ic[1], erro_padrao

        variaveis_ic = [
            ("Balance", "💰 Saldo Médio (€)"),
            ("Age", "🎂 Idade Média"),
            ("CreditScore", "📊 Score de Crédito Médio"),
            ("EstimatedSalary", "💵 Salário Estimado Médio"),
        ]

        ic_results = []
        for var, label in variaveis_ic:
            media, ic_inf, ic_sup, se = calc_ic(df_filtrado[var])
            ic_results.append(
                {
                    "Variável": label,
                    "Média Amostral": media,
                    "IC Inferior (2.5%)": ic_inf,
                    "IC Superior (97.5%)": ic_sup,
                    "Erro Padrão": se,
                    "Margem de Erro": ic_sup - media,
                }
            )

        ic_c1, ic_c2 = st.columns(2)

        for i, r in enumerate(ic_results):
            target_col = ic_c1 if i % 2 == 0 else ic_c2
            with target_col:
                st.markdown(
                    f"""
<div class="info-box">
    <strong>{r['Variável']}</strong><br>
    Média amostral: <strong>{r['Média Amostral']:,.2f}</strong><br>
    IC 95%: [<strong>{r['IC Inferior (2.5%)']:,.2f}</strong> ;
    <strong>{r['IC Superior (97.5%)']:,.2f}</strong>]<br>
    Margem de erro: ±{r['Margem de Erro']:,.2f}<br>
    <em>Interpretação: Com 95% de confiança, a verdadeira média populacional
    de {r['Variável'].split(' ', 1)[1] if ' ' in r['Variável'] else r['Variável']}
    está entre {r['IC Inferior (2.5%)']:,.2f} e {r['IC Superior (97.5%)']:,.2f}.</em>
</div>
""",
                    unsafe_allow_html=True,
                )

        # Visualização
        st.markdown("#### Visualização dos Intervalos de Confiança")
        df_ic = pd.DataFrame(ic_results)
        fig_ic = go.Figure()
        for i, row in df_ic.iterrows():
            fig_ic.add_trace(
                go.Scatter(
                    x=[row["IC Inferior (2.5%)"], row["Média Amostral"], row["IC Superior (97.5%)"]],
                    y=[row["Variável"]] * 3,
                    mode="markers+lines",
                    marker=dict(
                        size=[10, 16, 10],
                        color=[CORES["info"], CORES["primaria"], CORES["info"]],
                    ),
                    line=dict(color=CORES["info"], width=3),
                    name=row["Variável"],
                    showlegend=False,
                )
            )
        fig_ic.update_layout(
            **PLOTLY_LAYOUT,
            title="Intervalos de Confiança de 95% para as Médias",
            xaxis_title="Valor",
            height=350,
        )
        st.plotly_chart(fig_ic, use_container_width=True)

# =================================================================
# ABA 8 — INSIGHTS E RECOMENDAÇÕES
# =================================================================
with tabs[7]:
    st.markdown("### 💡 Insights Automáticos & Recomendações")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if total_clientes < 10:
        st.warning("Dados insuficientes para gerar insights com os filtros atuais.")
    else:
        # ── Gerar insights baseados nos dados filtrados ──

        saiu_ins = df_filtrado[df_filtrado["Exited"] == 1]
        ficou_ins = df_filtrado[df_filtrado["Exited"] == 0]

        insights = []

        # Insight 1: Idade
        if len(saiu_ins) > 0 and len(ficou_ins) > 0:
            media_idade_saiu = saiu_ins["Age"].mean()
            media_idade_ficou = ficou_ins["Age"].mean()
            if media_idade_saiu > media_idade_ficou:
                insights.append(
                    f"🎂 **Clientes mais velhos têm maior propensão ao churn.** "
                    f"Idade média de quem saiu: **{media_idade_saiu:.1f} anos** vs. "
                    f"quem ficou: **{media_idade_ficou:.1f} anos** "
                    f"(diferença de {media_idade_saiu - media_idade_ficou:.1f} anos)."
                )

        # Insight 2: Saldo
        if len(saiu_ins) > 0 and len(ficou_ins) > 0:
            media_saldo_saiu = saiu_ins["Balance"].mean()
            media_saldo_ficou = ficou_ins["Balance"].mean()
            if media_saldo_saiu > media_saldo_ficou:
                insights.append(
                    f"💰 **Clientes com maior saldo apresentam maior probabilidade de churn.** "
                    f"Saldo médio de quem saiu: **€ {media_saldo_saiu:,.0f}** vs. "
                    f"quem ficou: **€ {media_saldo_ficou:,.0f}**."
                )
            else:
                insights.append(
                    f"💰 **Clientes que saíram possuem saldo médio menor** "
                    f"(€ {media_saldo_saiu:,.0f}) comparado a quem ficou "
                    f"(€ {media_saldo_ficou:,.0f})."
                )

        # Insight 3: Geografia
        if len(df_filtrado["Geography"].unique()) > 1:
            churn_por_pais = df_filtrado.groupby("Geography")["Exited"].mean()
            pais_pior = churn_por_pais.idxmax()
            taxa_pior = churn_por_pais.max() * 100
            pais_melhor = churn_por_pais.idxmin()
            taxa_melhor = churn_por_pais.min() * 100
            insights.append(
                f"🌍 **{pais_pior} lidera em churn** com taxa de **{taxa_pior:.1f}%**, "
                f"enquanto **{pais_melhor}** possui a menor taxa (**{taxa_melhor:.1f}%**). "
                f"Diferença de {taxa_pior - taxa_melhor:.1f} pontos percentuais."
            )

        # Insight 4: Gênero
        if len(df_filtrado["Gender"].unique()) > 1:
            churn_genero = df_filtrado.groupby("Gender")["Exited"].mean() * 100
            if "Female" in churn_genero.index and "Male" in churn_genero.index:
                fem_rate = churn_genero["Female"]
                mal_rate = churn_genero["Male"]
                if fem_rate > mal_rate:
                    insights.append(
                        f"👩 **Mulheres apresentam maior taxa de churn** "
                        f"(**{fem_rate:.1f}%** vs. **{mal_rate:.1f}%** dos homens), "
                        f"uma diferença de {fem_rate - mal_rate:.1f} p.p."
                    )

        # Insight 5: Membro ativo
        if len(saiu_ins) > 0 and len(ficou_ins) > 0:
            ativo_saiu = saiu_ins["IsActiveMember"].mean() * 100
            ativo_ficou = ficou_ins["IsActiveMember"].mean() * 100
            if ativo_ficou > ativo_saiu:
                insights.append(
                    f"✅ **Ser membro ativo é fator protetor contra o churn.** "
                    f"Entre quem ficou, **{ativo_ficou:.1f}%** são ativos, "
                    f"contra apenas **{ativo_saiu:.1f}%** entre quem saiu."
                )

        # Insight 6: Número de produtos
        if len(df_filtrado) > 0:
            churn_prod = df_filtrado.groupby("NumOfProducts")["Exited"].mean() * 100
            if len(churn_prod) > 2 and churn_prod.get(3, 0) > 50:
                insights.append(
                    f"📦 **Clientes com 3+ produtos têm churn extremamente alto** "
                    f"(~{churn_prod.get(3, 0):.0f}%). "
                    f"Isso pode indicar insatisfação ou má experiência com múltiplos serviços."
                )

        # Exibir insights
        st.markdown("#### 📊 Insights Baseados nos Dados")
        for ins in insights:
            st.markdown(ins)
            st.markdown("")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Recomendações práticas ──
        st.markdown("#### 🎯 Recomendações Estratégicas para Retenção")

        recomendacoes = [
            {
                "titulo": "🎯 Segmentação de Risco Prioritário",
                "descricao": (
                    "Criar um **segmento de alto risco** com clientes que atendam a pelo menos "
                    "2 dos seguintes critérios: idade acima de 40 anos, saldo acima de "
                    "€100.000, residência na Alemanha, membro inativo. "
                    "Estes clientes devem receber atenção prioritária."
                ),
            },
            {
                "titulo": "📞 Programa de Contato Proativo",
                "descricao": (
                    "Implementar um programa de **contato proativo** para clientes de alto risco. "
                    "Ligar antes que decidam sair, oferecer revisão de portfólio personalizada "
                    "e benefícios exclusivos. Foco especial em clientes inativos há mais de 3 meses."
                ),
            },
            {
                "titulo": "🎁 Benefícios Personalizados por Perfil",
                "descricao": (
                    "**Mulheres 40+** na Alemanha: oferecer consultoria financeira gratuita e "
                    "taxas preferenciais. **Clientes com alto saldo**: programa de fidelidade "
                    "premium com cashback progressivo e acesso a investimentos exclusivos."
                ),
            },
            {
                "titulo": "🔄 Reativação de Membros Inativos",
                "descricao": (
                    "Criar campanhas de **reativação** com incentivos: isenção de tarifas por "
                    "6 meses, bônus por transações realizadas, acesso a funcionalidades premium "
                    "do app. Clientes inativos têm quase o dobro de churn comparado aos ativos."
                ),
            },
            {
                "titulo": "📦 Revisão da Estratégia de Produtos",
                "descricao": (
                    "Clientes com 3+ produtos apresentam churn crítico. **Simplificar o portfólio**, "
                    "consolidar produtos e garantir que cada produto agregue valor percebido. "
                    "Evitar cross-sell agressivo que gere fadiga no cliente."
                ),
            },
            {
                "titulo": "🌍 Estratégia Regional — Foco Alemanha",
                "descricao": (
                    "A Alemanha apresenta taxa de churn significativamente superior. "
                    "Investigar **fatores locais** (concorrência, satisfação, cultura bancária) "
                    "e adaptar produto/preços ao mercado alemão. Considerar pesquisa qualitativa "
                    "para entender as razões específicas."
                ),
            },
        ]

        for rec in recomendacoes:
            st.markdown(
                f"""
<div class="success-box">
    <strong>{rec['titulo']}</strong><br>
    {rec['descricao']}
</div>
""",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────
# Rodapé
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
<div style="text-align: center; color: #636e72; padding: 20px 0 10px 0; font-size: 0.85rem;">
    📊 Dashboard de Churn Bancário — Projeto Acadêmico de Ciência de Dados<br>
    Desenvolvido com Streamlit, Plotly, Scikit-learn e Scipy
</div>
""",
    unsafe_allow_html=True,
)
