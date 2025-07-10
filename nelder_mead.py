import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==============================================================================
# LÓGICA MATEMÁTICA DEL ALGORITMO
# ==============================================================================

def createSimplex(x0: np.ndarray, alpha: float, N: int):
    """Crea el simplex inicial."""
    delta1 = ((np.sqrt(N + 1) + N - 1) / (N * np.sqrt(2))) * alpha
    delta2 = ((np.sqrt(N + 1) - 1) / (N * np.sqrt(2))) * alpha
    
    # Crea los vértices
    simplex = [x0]
    for j in range(N):
        vertex = np.copy(x0).astype(float)
        for i in range(N):
            if i == j:
                vertex[i] += delta1
            else:
                vertex[i] += delta2
        simplex.append(vertex)
        
    return np.array(simplex)

# Funciones de operaciones del simplex (lambdas)
centroide_calcular = lambda simplex, index_mas_alto, N: (np.sum(simplex, axis=0) - simplex[index_mas_alto]) / N
reflexion = lambda centroide, mas_alto: 2 * centroide - mas_alto
expansion = lambda centroide, mas_alto, gamma: (1 + gamma) * centroide - gamma * mas_alto
contraccion_externa = lambda centroide, mas_alto, beta: (1 - beta) * centroide + beta * mas_alto
contraccion_interna = lambda centroide, mas_alto, beta: (1 + beta) * centroide - beta * mas_alto
terminar = lambda valores_funcion, fcentroide, N, epsilon: np.sqrt(np.sum((valores_funcion - fcentroide)**2) / (N + 1)) <= epsilon

# Función de encogimiento (cuando una contracción falla)
def encoger_simplex(simplex, i_xl, beta):
    xl = simplex[i_xl]
    for i in range(len(simplex)):
        if i != i_xl:
            simplex[i] = xl + beta * (simplex[i] - xl)
    return simplex

# ==============================================================================
# FUNCIÓN PRINCIPAL DEL ALGORITMO 
# ==============================================================================

def nelder_mead_algorithm(objective_func, x0, alpha, gamma, beta, epsilon, max_iter):
    """
    Ejecuta el algoritmo de Nelder-Mead.
    Devuelve el historial de pasos y los datos para la tabla de iteraciones.
    """
    N = len(x0)  # Número de dimensiones
    simplex = createSimplex(x0, alpha, N)
    
    history = {'simplex': [np.copy(simplex)], 'operation': ['Inicio']}
    table_data = []

    for k in range(max_iter):
        valores_funcion = np.array([objective_func(x) for x in simplex])
        indices = np.argsort(valores_funcion)
        i_xl, i_xg, i_mas_alto = indices[0], indices[-2], indices[-1]
        
        xl, xg, xh = simplex[i_xl], simplex[i_xg], simplex[i_mas_alto]
        fl, fg, fh = valores_funcion[i_xl], valores_funcion[i_xg], valores_funcion[i_mas_alto]
        
        # 1. Criterio de terminación
        centroide = centroide_calcular(simplex, i_mas_alto, N)
        fcentroide = objective_func(centroide)
        if terminar(valores_funcion, fcentroide, N, epsilon):
            history['operation'].append('Terminado')
            break
            
        # 2. Reflexión
        xr = reflexion(centroide, xh)
        fxr = objective_func(xr)
        operation = "Reflexión"

        # 3. Expansión
        if fxr < fl:
            xe = expansion(centroide, xh, gamma)
            fxe = objective_func(xe)
            if fxe < fxr:
                simplex[i_mas_alto] = xe
                operation = "Expansión"
            else:
                simplex[i_mas_alto] = xr
                operation = "Reflexión"
        # 4. Contracción
        elif fxr >= fh:
            xc = contraccion_externa(centroide, xh, beta)
            fxc = objective_func(xc)
            if fxc < fh:
                simplex[i_mas_alto] = xc
                operation = "Contracción Externa"
            else:
                # Encogimiento
                simplex = encoger_simplex(simplex, i_xl, beta)
                operation = "Encogimiento (Shrink)"
        # Reflexión Aceptada
        else: # fl <= fxr < fh
             simplex[i_mas_alto] = xr
             operation = "Reflexión Aceptada"

        history['simplex'].append(np.copy(simplex))
        history['operation'].append(operation)
        table_data.append({
            'Iteración': k + 1,
            'Mejor Valor f(x)': fl,
            'Peor Valor f(x)': fh,
            'Operación': operation,
            'Mejor Vértice': f"[{xl[0]:.3f}, {xl[1]:.3f}]"
        })
        
    return simplex[i_xl], valores_funcion[i_xl], history, table_data

# ==============================================================================
# INTERFAZ DE STREAMLIT
# ==============================================================================

def plot_contour_with_simplex(ax, fig, func, bounds, simplex, title, operation=""):
    """Dibuja el mapa de contorno y un simplex específico."""
    min_b, max_b = [b[0] for b in bounds], [b[1] for b in bounds]
    x_vals = np.linspace(min_b[0], max_b[0], 100)
    y_vals = np.linspace(min_b[1], max_b[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.array([func(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

    ax.clear()
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
    # Dibuja el simplex como un polígono cerrado
    simplex_closed = np.vstack([simplex, simplex[0]])
    ax.plot(simplex_closed[:, 0], simplex_closed[:, 1], 'r-o', markersize=5, linewidth=2, label='Simplex')
    ax.plot(simplex[0, 0], simplex[0, 1], 'yo', markersize=7, label='Mejor Vértice') # Mejor
    ax.plot(simplex[-1, 0], simplex[-1, 1], 'ko', markersize=7, label='Peor Vértice') # Peor
    
    ax.set_title(f"{title}\nOperación: {operation}")
    ax.set_xlabel('x_0')
    ax.set_ylabel('x_1')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig, ax

def show_nelder_mead(FUNCIONES, evaluar_funcion):
    st.markdown("## 🔶 Nelder-Mead Simplex")
    st.markdown("""
    Este método de optimización, también conocido como "simplex cuesta abajo", busca el mínimo de una función en un espacio multidimensional. Utiliza un poliedro llamado **simplex**, que se mueve por el espacio de la función, reflejándose, expandiéndose o contrayéndose para encontrar zonas con valores más bajos.
    - **Reflexión**: Intenta moverse en la dirección opuesta al peor punto.
    - **Expansión**: Si la reflexión fue buena, intenta alargar el paso en esa dirección.
    - **Contracción**: Si la reflexión fue mala, intenta acortar el paso.
    - **Encogimiento**: Como último recurso, reduce el tamaño del simplex completo hacia el mejor punto.
    """)

    # --- Configuración de parámetros ---
    with st.expander("Configuración del Algoritmo y la Función", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            funcion_seleccionada = st.selectbox("🎯 Selecciona la función:", list(FUNCIONES.keys()))
            info_funcion = FUNCIONES[funcion_seleccionada]
            
            st.markdown("**Parámetros del Simplex:**")
            alpha = st.slider("α (Reflexión)", 0.1, 2.0, 1.0, 0.1)
            gamma = st.slider("γ (Expansión)", 1.1, 3.0, 2.0, 0.1)
            beta = st.slider("β (Contracción/Encogimiento)", 0.1, 0.9, 0.5, 0.1)

        with col2:
            st.latex(info_funcion['latex'])
            st.markdown("**Parámetros de Ejecución:**")
            max_iter = st.slider("🔄 Número máximo de iteraciones:", 10, 500, 100, 10)
            epsilon = st.number_input("ε (Tolerancia de parada)", 0.0, 1.0, 0.001, format="%.5f")
            
            st.markdown("**📍 Punto inicial (x₀):**")
            bounds_list = info_funcion["intervalos"]
            ui_cols = st.columns(len(bounds_list))
            x0_vals = [
                col.number_input(f'x{i}', value=np.mean(bounds_list[i]), key=f'nm_x0_{i}')
                for i, col in enumerate(ui_cols)
            ]
            x0 = np.array(x0_vals)

    # --- Ejecución del algoritmo ---
    if st.button("🚀 Ejecutar Nelder-Mead", key="run_nelder_mead"):
        func_objetivo = lambda x: evaluar_funcion(x, funcion_seleccionada)
        
        with st.spinner("Optimizando con Nelder-Mead..."):
            x_best, f_best, history, table_data = nelder_mead_algorithm(
                func_objetivo, x0, alpha, gamma, beta, epsilon, max_iter
            )
        
        st.success("✅ Optimización completada!")
        
        col1, col2 = st.columns(2)
        col1.metric("🎯 Mejor solución (x_best)", f"[{x_best[0]:.4f}, {x_best[1]:.4f}]")
        col2.metric("📊 Valor óptimo f(x_best)", f"{f_best:.6f}")

        # --- Visualización ---
        st.markdown("### 📈 Visualización de la Optimización")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Gráfica 1: Simplex Inicial
        plot_contour_with_simplex(ax1, fig, func_objetivo, info_funcion["intervalos"], history['simplex'][0], "Simplex Inicial")
        
        # Gráfica 2: Simplex Final
        plot_contour_with_simplex(ax2, fig, func_objetivo, info_funcion["intervalos"], history['simplex'][-1], "Simplex Final")
        
        st.pyplot(fig)

        # --- Visualización interactiva ---
        st.markdown("#### 🔄 Visualizador de Iteraciones")
        iter_to_show = st.slider("Selecciona una iteración para visualizar:", 0, len(history['simplex'])-1, len(history['simplex'])-1)
        
        fig_iter, ax_iter = plt.subplots(figsize=(8, 7))
        plot_contour_with_simplex(
            ax_iter, fig_iter, func_objetivo, info_funcion["intervalos"], 
            history['simplex'][iter_to_show], 
            f"Simplex en Iteración {iter_to_show}",
            history['operation'][iter_to_show]
        )
        st.pyplot(fig_iter)
        
        # --- Tabla de resultados ---
        st.markdown("### 📋 Tabla de Iteraciones")
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        # --- Contexto y Aplicaciones ---
        st.markdown("---")
        st.subheader("Contexto y Aplicaciones")
        with st.expander("Acerca del Algoritmo Nelder-Mead"):
            st.markdown("""
            El **algoritmo Nelder-Mead**, también conocido como método del simplex cuesta abajo, es un método de optimización sin derivadas que busca minimizar una función objetivo en un espacio multidimensional. Utiliza un **simplex**, un poliedro con \(n+1\) vértices en un espacio de \(n\) dimensiones (por ejemplo, un triángulo en 2D), que se transforma iterativamente mediante operaciones como reflexión, expansión, contracción y encogimiento para acercarse al mínimo de la función.

            ### ¿Cómo funciona?
            - **Inicialización**: Se crea un simplex inicial a partir de un punto inicial \(x_0\), utilizando parámetros como \(\alpha\) para definir su tamaño.
            - **Operaciones del Simplex**:
              - **Reflexión**: Mueve el peor vértice (con el mayor valor de la función) hacia el lado opuesto del centroide de los otros vértices.
              - **Expansión**: Si la reflexión produce un punto mejor que el mejor vértice actual, intenta extender el movimiento para explorar más allá.
              - **Contracción**: Si la reflexión no mejora lo suficiente, reduce el paso hacia el centroide.
              - **Encogimiento**: Si ninguna operación mejora el simplex, lo reduce hacia el mejor vértice.
            - **Criterio de parada**: El algoritmo termina cuando la desviación estándar de los valores de la función en los vértices es menor que una tolerancia \(\epsilon\), o se alcanza el número máximo de iteraciones.

            ### Aplicaciones
            - **Optimización no lineal**: Ideal para funciones donde las derivadas no están disponibles o son difíciles de calcular.
            - **Aprendizaje automático**: Usado para optimizar hiperparámetros o funciones de pérdida en modelos donde el cálculo del gradiente es costoso.
            - **Ingeniería y ciencias**: Aplicado en ajuste de modelos, optimización de diseños y simulaciones físicas (por ejemplo, ajuste de parámetros en modelos químicos o mecánicos).
            - **Problemas de baja dimensión**: Eficaz en espacios de dimensión moderada (generalmente \(n < 10\)).

            ### Ventajas
            - No requiere cálculo de derivadas, lo que lo hace adecuado para funciones no diferenciables o ruidosas.
            - Simple de implementar y entender.
            - Robusto para problemas con mínimos locales en algunos casos.

            ### Limitaciones
            - Puede converger lentamente o estancarse en mínimos locales, especialmente en funciones con muchos mínimos.
            - Sensible a la elección del simplex inicial (tamaño y posición).
            - Menos eficiente en espacios de alta dimensión en comparación con métodos basados en gradientes.
            - La convergencia no está garantizada para funciones no convexas o con discontinuidades.

            Este código ofrece una implementación educativa del algoritmo Nelder-Mead, con visualizaciones interactivas que muestran cómo el simplex evoluciona en un espacio 2D, facilitando la comprensión de su comportamiento y dinámica.
            """)