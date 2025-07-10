import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def busqueda_seccion_dorada(f, a, b, tol=1e-7, max_iter=100):
    """Tu implementación de la búsqueda de sección dorada para la búsqueda lineal."""
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi
    c = a + resphi * (b - a)
    d = b - resphi * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if abs(d - c) < tol:
            return (c + d) / 2
        if fc < fd:
            b, d, fd, c, fc = d, c, fc, a + resphi * (b - a), f(a + resphi * (b - a))
        else:
            a, c, fc, d, fd = c, d, fd, b - resphi * (b - a), f(b - resphi * (b - a))
    return (a + b) / 2

def conjugate_gradient_algorithm(f, grad_f, x0, tol1, tol2, max_iter):
    k = 0
    x = np.array(x0, dtype=float)
    g = grad_f(x)
    d = -g
    epsilon = 1e-12
    path = [x.copy()]
    table_data = []

    while np.linalg.norm(g) > tol1 and k < max_iter:
        line_search_func = lambda alpha: f(x + alpha * d)
        alpha_k = busqueda_seccion_dorada(line_search_func, a=0.0, b=1.0, tol=tol2)
        
        x_new = x + alpha_k * d
        path.append(x_new.copy())
        
        table_data.append({
            'Iteración (k)': k,
            'xk': f"[{x[0]:.4f}, {x[1]:.4f}]",
            'f(xk)': f(x),
            '||∇f(xk)||': np.linalg.norm(g),
            'alpha*': alpha_k,
        })
        
        if np.linalg.norm(x_new - x) / (np.linalg.norm(x) + epsilon) < tol2:
            x = x_new
            break
            
        g_new = grad_f(x_new)
        beta_k_plus_1 = np.dot(g_new, g_new) / (np.dot(g, g) + epsilon)
        d = -g_new + beta_k_plus_1 * d
        g, x, k = g_new, x_new, k + 1
        
    return x, f(x), k, np.array(path), table_data

def plot_cg_results(ax, f, path, intervals):
    x_min, x_max = intervals[0]; y_min, y_max = intervals[1]
    x_grid = np.linspace(x_min, x_max, 200)
    y_grid = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.array([f(p) for p in np.stack([X.ravel(), Y.ravel()], axis=-1)]).reshape(X.shape)

    min_z, max_z = np.min(Z), np.max(Z)
    levels = np.logspace(np.log10(min_z if min_z > 0 else 1e-3), np.log10(max_z if max_z > 0 else 1), 20)
    
    ax.contourf(X, Y, Z, levels=levels, cmap='viridis_r', alpha=0.85)
    ax.plot(path[:, 0], path[:, 1], 'r-o', markersize=4, linewidth=2, label='Trayectoria')
    ax.scatter(path[0, 0], path[0, 1], c='cyan', marker='P', s=200, zorder=3, label='Inicio')
    ax.scatter(path[-1, 0], path[-1, 1], c='lime', marker='*', s=250, zorder=3, label='Mínimo')
    
    ax.set_title("Optimización con Gradiente Conjugado", fontsize=16)
    ax.set_xlabel('x₀'); ax.set_ylabel('x₁')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

def show_conjugate_gradient(FUNCIONES):
    st.markdown("## 📐 Gradiente Conjugado")
    st.markdown("""
    El método del Gradiente Conjugado (Fórmula de Fletcher-Reeves) es una mejora sofisticada sobre el descenso de gradiente. En lugar de moverse siempre en la dirección del gradiente local, construye una secuencia de direcciones de búsqueda que son "conjugadas".
    
    Esto le permite "recordar" las direcciones anteriores para no deshacer el progreso ya hecho, evitando el comportamiento en zigzag del método de Cauchy y convergiendo mucho más rápido, especialmente en valles largos y estrechos.
    
    La dirección de búsqueda `d` se actualiza con un factor `β` que incorpora información del gradiente anterior: `d_k = -∇f(x_k) + β_k * d_{k-1}`.
    """)

    with st.expander("Configuración del Algoritmo y la Función", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            funcion_seleccionada = st.selectbox("🎯 Selecciona la función:", list(FUNCIONES.keys()), key="cg_func")
            info_funcion = FUNCIONES[funcion_seleccionada]
            
            st.markdown("**Parámetros de Parada (Tolerancias):**")
            tol1 = st.number_input("ε₁ (Tolerancia de ||∇f||)", 0.0, 1.0, 1e-6, format="%.7f")
            tol2 = st.number_input("ε₂ (Tolerancia de paso)", 0.0, 1.0, 1e-6, format="%.7f")

        with col2:
            st.latex(info_funcion.get("latex", ""))
            st.markdown("**Parámetros de Ejecución:**")
            max_iter = st.slider("M (Máximo de iteraciones)", 10, 2000, 1000, 20)
            
            st.markdown("**📍 Punto Inicial (x₀):**")
            intervalos = info_funcion["intervalos"]
            x0_vals = [st.number_input(f'x0_{i}', value=np.random.uniform(inter[0], inter[1]), key=f'cg_x0_{i}') for i, inter in enumerate(intervalos)]
            x0 = np.array(x0_vals)

    if st.button("🚀 Ejecutar Gradiente Conjugado", key="run_cg"):
        f = info_funcion["funcion"]
        grad_f = info_funcion.get("gradiente")
        
        if not grad_f:
            st.error(f"Error: La función '{funcion_seleccionada}' no tiene un gradiente analítico definido en el diccionario. Este método lo requiere.")
            return

        with st.spinner("Optimizando con el método del Gradiente Conjugado..."):
            min_x, min_f, iteraciones, path, table_data = conjugate_gradient_algorithm(
                f, grad_f, x0, tol1, tol2, max_iter
            )
        
        st.success("✅ Optimización completada!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Solución Encontrada (x*)", f"[{min_x[0]:.5f}, {min_x[1]:.5f}]")
        col2.metric("📊 Valor Óptimo f(x*)", f"{min_f:.6f}")
        col3.metric("🔄 Iteraciones", f"{iteraciones}")

        st.markdown("### 📈 Visualización de la Optimización")
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_cg_results(ax, f, path, info_funcion["intervalos"])
        st.pyplot(fig)
        
        st.markdown("### 📋 Tabla de Iteraciones")
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        # 🔹 Información adicional basada en las diapositivas
        st.markdown("### 📘 Información adicional: Método del Gradiente Conjugado")
        st.markdown("""
        El **Método del Gradiente Conjugado** pertenece a la categoría de métodos **basados en gradiente** para optimización numérica. A diferencia del descenso de gradiente simple, este método utiliza direcciones de búsqueda conjugadas para evitar oscilaciones y acelerar la convergencia.

        🔍 **Características clave:**
        - Diseñado principalmente para minimizar funciones **cuadráticas** o suavemente curvadas.
        - Se basa en la fórmula: $\\vec{d}_k = -\\nabla f(x_k) + \\beta_k \\vec{d}_{k-1}$ con $\\beta_k$ de Fletcher-Reeves.
        - Utiliza **búsqueda unidireccional** (como sección dorada) para determinar $\\alpha_k$.

        🧠 **Fundamento teórico:**
        - Forma parte de los **métodos numéricos iterativos**.
        - Mejora los métodos de descenso simple, evitando zigzags en valles estrechos.
        - Muy eficiente en problemas con muchas variables donde calcular la Hessiana sería costoso.

        📎 **Ventajas y desventajas:**
        - ✅ No requiere almacenar ni invertir matrices grandes.
        - ✅ Converge más rápido que el gradiente simple en muchos casos.
        - ❌ Puede perder eficiencia si la función no es bien condicionada.
        - ❌ Requiere gradiente exacto o muy bien estimado.

        📚 Basado en las diapositivas de “Optimización - clase introductoria” (Adán E. Aguilar) y el libro de Kalyanmoy Deb.
        """)
