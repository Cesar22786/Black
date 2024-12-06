import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# Configuración de la página
st.set_page_config(
    page_title="Análisis Avanzado de Portafolios",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== ESTILOS Y GIF ====== #
def agregar_fondo_y_gif():
    # Fondo y estilos
    fondo_html = """
    <style>
    body {
        background: linear-gradient(to right, #000000, #800000);
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background: linear-gradient(to bottom, #000000, #800000);
    }
    .stSidebar {
        background: linear-gradient(to bottom, #400000, #800000);
    }
    .metric-container .metric {
        background: #400000;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
    }
    .stMarkdown h1, h2, h3 {
        color: #FFD700;
    }
    .css-1q8dd3e p {
        color: #ffcccc !important;
    }
    .dataframe {
        color: #ffffff !important;
        background-color: #333333;
    }
    .gif-container {
        position: absolute;
        top: 10px;
        right: 10px;
        width: 150px;
        z-index: 10;
    }
    </style>
    <div class="gif-container">
        <img src="https://art.pixilart.com/8ebf216d8b6c2f3.gif" alt="GIF" style="width:100%;border-radius:10px;box-shadow:0px 4px 10px rgba(0,0,0,0.5);">
    </div>
    """
    st.markdown(fondo_html, unsafe_allow_html=True)

# Llamar a la función para agregar el fondo y el GIF
agregar_fondo_y_gif()

# Guardar datos en CSV
def guardar_datos_csv(data, filename):
    data.to_csv(filename)
    st.success(f"Datos guardados en {filename}.")

# Descargar datos de Yahoo Finance
def descargar_datos(etfs, start_date, end_date):
    with st.spinner(f"Descargando datos desde {start_date} hasta {end_date}..."):
        try:
            data = yf.download(etfs, start=start_date, end=end_date)['Adj Close']
            if data.empty:
                st.error("No se pudieron descargar datos para los símbolos ingresados. Verifique las fechas o los símbolos.")
            return data.ffill().dropna()
        except Exception as e:
            st.error(f"Error al descargar datos: {e}")
            return pd.DataFrame()

# Calcular métricas básicas de los activos
def calcular_metricas(data):
    rendimientos = data.pct_change().dropna()
    media = rendimientos.mean() * 252
    volatilidad = rendimientos.std() * np.sqrt(252)
    sharpe = media / volatilidad
    sortino = media / (rendimientos[rendimientos < 0].std() * np.sqrt(252))
    drawdown = (data / data.cummax() - 1).min()
    return rendimientos, media, volatilidad, sharpe, sortino, drawdown

# Optimizar portafolios
def optimizar_portafolio(rendimientos, weights, tasa_libre_riesgo=0.02):
    n = len(rendimientos.columns)
    mean_returns = rendimientos.mean() * 252
    cov_matrix = rendimientos.cov() * 252

    def port_metrics(weights):
        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - tasa_libre_riesgo) / port_volatility
        return -sharpe

    bounds = [(0, 1) for _ in range(n)]
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    result = minimize(port_metrics, weights, bounds=bounds, constraints=constraints)
    return result.x, mean_returns, cov_matrix

# Modelo Black-Litterman
def black_litterman(mean_returns, cov_matrix, market_weights, views, confidence):
    try:
        tau = 0.05  # Parámetro de escala
        pi = np.dot(cov_matrix, market_weights)  # Retornos implícitos del mercado

        # Validación de vistas
        if len(views) != len(market_weights):
            raise ValueError("El número de vistas no coincide con el número de activos seleccionados.")

        Q = np.array(views).reshape(-1, 1)  # Vistas expresadas como matriz columna
        P = np.eye(len(market_weights))  # Matriz identidad (1 vista por activo)

        omega = np.diag(np.diag(np.dot(P, np.dot(tau * cov_matrix, P.T))) / confidence)  # Matriz de incertidumbre

        M_inverse = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(P.T, np.dot(np.linalg.inv(omega), P)))
        BL_returns = M_inverse @ (np.linalg.inv(tau * cov_matrix) @ pi + P.T @ np.linalg.inv(omega) @ Q)
        return BL_returns.flatten()

    except Exception as e:
        st.error(f"Error en el modelo Black-Litterman: {e}")
        return []

# ====== Entrada de Parámetros del Usuario ====== #
st.sidebar.header("Parámetros del Portafolio")
etfs_input = st.sidebar.text_input("Ingrese los ETFs separados por comas:", "AGG,EMB,VTI,EEM,GLD")
etfs = [etf.strip() for etf in etfs_input.split(',')]

start_date = st.sidebar.date_input("Fecha de inicio:", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("Fecha de fin:", pd.to_datetime("2023-01-01"))

weights_input = st.sidebar.text_input("Pesos iniciales (opcional):", ",".join(["0.2"] * len(etfs)))
weights = [float(w.strip()) for w in weights_input.split(",")] if weights_input else [1 / len(etfs)] * len(etfs)

# Descargar datos
data = descargar_datos(etfs, start_date, end_date)
rendimientos, media, volatilidad, sharpe, sortino, drawdown = calcular_metricas(data)

# Métricas visuales
st.title("📊 Análisis del Portafolio")
st.metric("Rendimiento Promedio Anualizado", f"{media.mean():.2%}")
st.metric("Volatilidad Promedio Anualizada", f"{volatilidad.mean():.2%}")

# Modelo Black-Litterman
st.header("🔮 Modelo Black-Litterman")
views_input = st.text_input("Ingrese las vistas (rendimientos esperados por activo):", "0.03,0.04,0.05,0.02,0.01")
confidence_input = st.slider("Confianza en las vistas (0-100):", 0, 100, 50)

views = [float(v.strip()) for v in views_input.split(",")]
confidence = confidence_input / 100
market_weights = np.array([1 / len(etfs)] * len(etfs))

bl_returns = black_litterman(media.values, rendimientos.cov(), market_weights, views, confidence)
st.write("Retornos ajustados por Black-Litterman:", pd.DataFrame(bl_returns, index=etfs, columns=["Rendimientos"]))


