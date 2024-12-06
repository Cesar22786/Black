import streamlit as st
import numpy as np
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Modelo Black-Litterman",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ====== Función Black-Litterman ====== #
def black_litterman(mean_returns, cov_matrix, market_weights, views, confidence):
    """
    Implementación del modelo de Black-Litterman.

    Parámetros:
        mean_returns (array): Retornos esperados del mercado.
        cov_matrix (DataFrame): Matriz de covarianza de los activos.
        market_weights (array): Pesos actuales del mercado (capitalización).
        views (array): Opiniones de los rendimientos esperados por el inversor.
        confidence (float): Nivel de confianza del inversor en sus opiniones (entre 0 y 1).

    Retorna:
        np.array: Retornos ajustados por Black-Litterman.
    """
    try:
        tau = 0.05  # Parámetro de escala para la incertidumbre
        pi = np.dot(cov_matrix, market_weights)  # Retornos implícitos del mercado

        P = np.eye(len(market_weights))  # Matriz identidad (1 vista por activo)
        Q = np.array(views).reshape(-1, 1)  # Vistas como matriz columna

        # Validación de dimensiones
        if P.shape[0] != Q.shape[0]:
            raise ValueError("Las dimensiones de la matriz P y las vistas Q no coinciden.")

        omega = np.diag(np.diag(np.dot(P, np.dot(tau * cov_matrix, P.T))) / confidence)  # Incertidumbre

        # Cálculo de los retornos ajustados
        M_inverse = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(P.T, np.dot(np.linalg.inv(omega), P)))
        BL_returns = M_inverse @ (np.linalg.inv(tau * cov_matrix) @ pi + P.T @ np.linalg.inv(omega) @ Q)

        return BL_returns.flatten()  # Retornar como arreglo unidimensional
    except Exception as e:
        st.error(f"Error en el modelo Black-Litterman: {e}")
        return None

# ====== Interfaz Streamlit ====== #
st.title("🔮 Modelo Black-Litterman")

# Entradas del usuario
st.sidebar.header("Parámetros del Modelo")

# Retornos esperados del mercado
mean_returns = st.sidebar.text_area(
    "Retornos esperados del mercado (separados por comas):",
    value="0.02,0.03,0.04,0.05,0.06"
)
mean_returns = np.array([float(x.strip()) for x in mean_returns.split(",")])

# Matriz de covarianza
cov_matrix_input = st.sidebar.text_area(
    "Matriz de covarianza (filas separadas por punto y coma, columnas por comas):",
    value="0.1,0.02,0.04,0.03,0.02;0.02,0.2,0.06,0.05,0.03;0.04,0.06,0.3,0.07,0.04;0.03,0.05,0.07,0.4,0.05;0.02,0.03,0.04,0.05,0.5"
)
cov_matrix = np.array([[float(y) for y in x.split(",")] for x in cov_matrix_input.split(";")])

# Pesos actuales del mercado
market_weights = st.sidebar.text_area(
    "Pesos del mercado (separados por comas):",
    value="0.2,0.2,0.2,0.2,0.2"
)
market_weights = np.array([float(x.strip()) for x in market_weights.split(",")])

# Opiniones del inversor
views = st.sidebar.text_area(
    "Vistas/opiniones del inversor (rendimientos esperados separados por comas):",
    value="0.025,0.035,0.045,0.055,0.065"
)
views = np.array([float(x.strip()) for x in views.split(",")])

# Confianza en las vistas
confidence = st.sidebar.slider("Confianza en las vistas (0-100):", min_value=0, max_value=100, value=80) / 100

# Botón para calcular
if st.sidebar.button("Calcular Retornos Ajustados"):
    bl_returns = black_litterman(mean_returns, cov_matrix, market_weights, views, confidence)

    if bl_returns is not None:
        # Mostrar resultados
        st.subheader("Resultados: Retornos Ajustados por Black-Litterman")
        resultados = pd.DataFrame({
            "ETF": [f"ETF {i+1}" for i in range(len(bl_returns))],
            "Retornos Ajustados": bl_returns
        })
        st.table(resultados)
