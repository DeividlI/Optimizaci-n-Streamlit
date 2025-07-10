import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

# --- Funciones Auxiliares (Búsqueda de Línea y Diferenciación Numérica) ---

def w_to_x(w: float, a, b) -> float:
    return w * (b - a) + a

def busquedaDorada(funcion, epsilon: float, a: float = 0.0, b: float = 1.0) -> float:
    PHI = (1 + math.sqrt(5)) / 2 - 1
    aw, bw = 0, 1
    Lw = 1
    max_iter_dorada = 500
    k = 1
    w2 = bw - PHI * Lw
    w1 = aw + PHI * Lw
    fx2 = funcion(w_to_x(w2, a, b))
    fx1 = funcion(w_to_x(w1, a, b))

    while Lw > epsilon and k < max_iter_dorada:
        if fx1 > fx2:
            bw, w1, fx1 = w1, w2, fx2
            Lw = bw - aw
            w2 = bw - PHI * Lw
            fx2 = funcion(w_to_x(w2, a, b))
        else:
            aw, w2, fx2 = w2, w1, fx1
            Lw = bw - aw
            w1 = aw + PHI * Lw
            fx1 = funcion(w_to_x(w1, a, b))
        k += 1
    return (w_to_x(aw, a, b) + w_to_x(bw, a, b)) / 2

def gradiente(f, x, deltaX=1e-5):
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += deltaX
        xn = x.copy(); xn[i] -= deltaX
        grad[i] = (f(xp) - f(xn)) / (2 * deltaX)
    return grad

def hessiano(f, x, deltaX=1e-4):
    n = len(x)
    hess = np.zeros((n, n), dtype=float)
    fx = f(x)
    for i in range(n):
        for j in range(n):
            if i == j:
                xp = x.copy(); xp[i] += deltaX
                xn = x.copy(); xn[i] -= deltaX
                hess[i, j] = (f(xp) - 2 * fx + f(xn)) / (deltaX**2)
            else:
                xpp = x.copy(); xpp[i] += deltaX; xpp[j] += deltaX
                xpn = x.copy(); xpn[i] += deltaX; xpn[j] -= deltaX
                xnp = x.copy(); xnp[i] -= deltaX; xnp[j] += deltaX
                xnn = x.copy(); xnn[i] -= deltaX; xnn[j] -= deltaX
                hess[i, j] = (f(xpp) - f(xpn) - f(xnp) + f(xnn)) / (4 * deltaX**2)
    return hess

def newton_algorithm(funcion, x0, epsilon1, epsilon2, M):
    xk = np.array(x0, dtype=float)
    historial = [xk]
    table_data = []
    termination_reason = "Se alcanzó el máximo número de iteraciones."

    for k in range(M):
        grad_k = gradiente(funcion, xk)
        grad_norm = np.linalg.norm(grad_k)

        if grad_norm < epsilon1:
            termination_reason = f"Convergencia por gradiente pequeño (||∇f|| < {epsilon1}) en la iteración {k}."
            break

        hess_k = hessiano(funcion, xk)
        try:
            regularization = 1e-8 * np.identity(len(xk))
            hess_inv_k = np.linalg.inv(hess_k + regularization)
        except np.linalg.LinAlgError:
            termination_reason = "Error: La matriz Hessiana es singular. No se puede continuar."
            break
            
        pk = -np.dot(hess_inv_k, grad_k)
        
        direccion = "Newton"
        if np.dot(pk, grad_k) > 0:
            pk = -grad_k
            direccion = "Descenso de Gradiente (fallback)"

        def alpha_funcion(alpha): return funcion(xk + alpha * pk)
        alpha_optimo = busquedaDorada(alpha_funcion, epsilon2)
        
        x_k1 = xk + alpha_optimo * pk
        historial.append(x_k1)
        
        table_data.append({
            'Iteración (k)': k,
            'xk': f"[{xk[0]:.4f}, {xk[1]:.4f}]",
            '||∇f(xk)||': grad_norm,
            'α*': alpha_optimo,
            'Dirección': direccion,
            'xk+1': f"[{x_k1[0]:.4f}, {x_k1[1]:.4f}]"
        })
        
        if np.linalg.norm(x_k1 - xk) / (np.linalg.norm(xk) + 1e-8) < epsilon2 and k > 0:
            termination_reason = f"Convergencia por cambio relativo pequeño en la iteración {k+1}."
            xk = x_k1
            break
            
        xk = x_k1
    
    return xk, historial, table_data, termination_reason

def plot_newton_results(ax, func, bounds, history):
    x_min, x_max = bounds[0]; y_min, y_max = bounds[1]
    rango_x = np.linspace(x_min, x_max, 200)
    rango_y = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(rango_x, rango_y)
    Z = func([X, Y])

    ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.9)
    ruta = np.array(history)
    ax.plot(ruta[:, 0], ruta[:, 1], 'o-', color='red', markersize=4, linewidth=1.5, label='Ruta de Optimización')
    ax.plot(ruta[0, 0], ruta[0, 1], 'o', color='lime', markersize=8, label='Inicio')
    ax.plot(ruta[-1, 0], ruta[-1, 1], '*', color='yellow', markersize=12, label='Fin (Solución)')
    
    ax.set_title("Ruta de Optimización con Método de Newton", fontsize=16)
    ax.set_xlabel('x_0'); ax.set_ylabel('x_1')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)

def show_newton_method(FUNCIONES, evaluar_funcion):
    st.markdown("## 🔢 Método de Newton")
    st.markdown("""
    El Método de Newton es un algoritmo de **segundo orden**, lo que significa que utiliza no solo el gradiente (primera derivada) sino también la **matriz Hessiana** (segunda derivada) para modelar la función como una superficie cuadrática en cada paso.
    """)

    with st.expander("Configuración del Algoritmo y la Función", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            funcion_seleccionada = st.selectbox("🎯 Selecciona la función:", list(FUNCIONES.keys()), key="newton_func")
            info_funcion = FUNCIONES[funcion_seleccionada]
            
            st.markdown("**Parámetros de Parada (Tolerancias):**")
            epsilon1 = st.number_input("ε₁ (Tolerancia de ||∇f||)", 0.0, 1.0, 1e-5, format="%.6f")
            epsilon2 = st.number_input("ε₂ (Tolerancia de paso)", 0.0, 1.0, 1e-5, format="%.6f")

        with col2:
            st.latex(info_funcion['latex'])
            st.markdown("**Parámetros de Ejecución:**")
            max_iter = st.slider("M (Máximo de iteraciones)", 10, 200, 100, 5)
            st.markdown("**📍 Punto Inicial (x₀):**")
            default_x0 = info_funcion.get("x0", np.array([0.0, 0.0]))
            x0_cols = st.columns(len(default_x0))
            x0_vals = [c.number_input(f'x0_{i}', value=float(default_x0[i]), key=f'newton_x0_{i}') for i, c in enumerate(x0_cols)]
            x0 = np.array(x0_vals)

    if st.button("🚀 Ejecutar Método de Newton", key="run_newton"):
        func_objetivo = lambda x: evaluar_funcion(x, funcion_seleccionada)
        
        with st.spinner("Calculando gradientes y hessianos..."):
            solucion, historial, table_data, reason = newton_algorithm(
                func_objetivo, x0, epsilon1, epsilon2, max_iter
            )
        
        st.success("✅ Optimización completada!")
        
        col1, col2 = st.columns(2)
        col1.metric("🎯 Solución Encontrada (x*)", f"[{solucion[0]:.5f}, {solucion[1]:.5f}]")
        col2.metric("📊 Valor Óptimo f(x*)", f"{func_objetivo(solucion):.6f}")
        st.info(f"Razón de la parada: {reason}")

        st.markdown("### 📈 Visualización de la Optimización")
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_newton_results(ax, func_objetivo, info_funcion["intervalos"], historial)
        st.pyplot(fig)
        
        st.markdown("### 📋 Tabla de Iteraciones")
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        st.markdown("### 📘 Información adicional: Método de Newton")
        st.markdown("""
        El **Método de Newton** es un algoritmo de optimización de segundo orden que utiliza derivadas tanto de primer orden (gradientes) como de segundo orden (Hessianas) para encontrar puntos óptimos (mínimos o máximos locales) de una función.

        🔍 **Resumen de sus características clave:**
        - Utiliza una **aproximación cuadrática** de la función para buscar el mínimo.
        - Calcula una dirección de búsqueda resolviendo: $H(x_k) p_k = -\\nabla f(x_k)$
        - Realiza una **búsqueda unidireccional** para encontrar el mejor paso α en esa dirección.
        - Tiene **alta velocidad de convergencia**, especialmente si el punto inicial está cerca del óptimo.
        - Puede fallar si la Hessiana es **singular o indefinida**, o si el punto inicial está lejos del mínimo.

        🧠 **Fundamento teórico (según tus diapositivas):**
        - Forma parte de los **métodos numéricos**, que construyen soluciones iterativas mejoradas.
        - Se apoya en los **métodos basados en gradiente**, que explotan la información derivada.
        - La Hessiana y el gradiente pueden calcularse **numéricamente** (por diferencia central).
        - Si el **determinante de la Hessiana es > 0** y $f_{xx}>0$, hay un mínimo local.

        📎 **Ventajas y desventajas:**
        - ✅ Muy eficiente en funciones suaves y bien condicionadas.
        - ❌ Computacionalmente costoso por la inversión de matrices.
        - ❌ No garantiza siempre el descenso, se puede necesitar fallback como el gradiente descendente.

        📖 **Aplicaciones típicas**: Ingeniería, Machine Learning, Diseño óptimo, Modelado de sistemas continuos.

        📚 Basado en: *Optimización - clase introductoria*, curso de Adán E. Aguilar y libro de Deb Kalyanmoy.
        """)
