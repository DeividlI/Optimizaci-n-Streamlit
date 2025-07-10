import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def busqueda_unidireccional_core(funcion_objetivo, x_t, s_t, alpha_inicio=0.0, paso_alpha=0.1, max_iteraciones=100):
    """
    Realiza una búsqueda unidireccional básica para encontrar el mínimo a lo largo de una línea.

    Args:
        funcion_objetivo (callable): La función a minimizar.
        x_t (np.array): El punto actual (x^(t)). Un arreglo numpy 1D.
        s_t (np.array): La dirección de búsqueda (s^(t)). Un arreglo numpy 1D.
        alpha_inicio (float): Valor inicial para alpha.
        paso_alpha (float): Tamaño del paso para alpha.
        max_iteraciones (int): Número máximo de iteraciones para la búsqueda.

    Returns:
        tuple: (x_optimo, alpha_optimo, historial, iter_info_pos, iter_info_neg)
            - x_optimo (np.array): El punto x(alpha) que minimiza la función objetivo.
            - alpha_optimo (float): El valor de alpha correspondiente al x_optimo.
            - historial (list): Lista de pares (alpha, valor_objetivo) durante la búsqueda.
            - iter_info_pos (list): Información de iteraciones en dirección positiva.
            - iter_info_neg (list): Información de iteraciones en dirección negativa.
    """
    if not isinstance(x_t, np.ndarray) or not isinstance(s_t, np.ndarray):
        st.error("Error: x_t y s_t deben ser arreglos numpy.")
        return None, None, [], [], []
    if x_t.shape != s_t.shape:
        st.error("Error: x_t y s_t deben tener las mismas dimensiones.")
        return None, None, [], [], []

    historial = []
    alpha_actual = alpha_inicio
    x_actual = x_t + alpha_actual * s_t
    valor_min_objetivo = funcion_objetivo(x_actual)
    x_optimo = x_actual.copy()
    alpha_optimo = alpha_actual
    historial.append((alpha_actual, valor_min_objetivo))

    with st.container():
        st.write(f"**Punto inicial** $x(\\alpha_{{inicial}})$: {np.round(x_actual, 4)}, **Valor Objetivo**: {valor_min_objetivo:.6f}")

    # Búsqueda en dirección positiva
    alpha_temp_pos = alpha_actual
    iter_info_pos = []
    for i in range(max_iteraciones):
        siguiente_alpha = alpha_temp_pos + paso_alpha
        siguiente_x = x_t + siguiente_alpha * s_t
        siguiente_valor_objetivo = funcion_objetivo(siguiente_x)
        
        historial.append((siguiente_alpha, siguiente_valor_objetivo))
        iter_info_pos.append((i+1, siguiente_alpha, np.round(siguiente_x, 4), siguiente_valor_objetivo))

        if siguiente_valor_objetivo >= valor_min_objetivo and i > 0:
            break
        
        valor_min_objetivo = siguiente_valor_objetivo
        alpha_optimo = siguiente_alpha
        x_optimo = siguiente_x.copy()
        alpha_temp_pos = siguiente_alpha

    # Búsqueda en dirección negativa
    alpha_temp_neg = alpha_inicio
    iter_info_neg = []
    for i in range(max_iteraciones):
        if alpha_temp_neg - paso_alpha < -100:
            break
        siguiente_alpha = alpha_temp_neg - paso_alpha
        siguiente_x = x_t + siguiente_alpha * s_t
        siguiente_valor_objetivo = funcion_objetivo(siguiente_x)
        
        historial.append((siguiente_alpha, siguiente_valor_objetivo))
        iter_info_neg.append((i+1, siguiente_alpha, np.round(siguiente_x, 4), siguiente_valor_objetivo))

        if siguiente_valor_objetivo < valor_min_objetivo:
            valor_min_objetivo = siguiente_valor_objetivo
            alpha_optimo = siguiente_alpha
            x_optimo = siguiente_x.copy()
        else:
            if i > 0:
                break
        alpha_temp_neg = siguiente_alpha

    historial.sort(key=lambda x: x[0])
    return x_optimo, alpha_optimo, historial, iter_info_pos, iter_info_neg

def show_busqueda_unidireccional(funciones_multivariadas, evaluar_funcion_multivariada):
    st.markdown('<h1 class="main-title">Búsqueda Unidireccional</h1>', unsafe_allow_html=True)
    st.markdown("""
    La búsqueda unidireccional encuentra el mínimo de una función multivariable a lo largo de una dirección específica $s^{(t)}$ desde un punto $x^{(t)}$.  
    Los puntos en la línea se expresan como: $x(\\alpha) = x^{(t)} + \\alpha s^{(t)}$, donde $\\alpha$ es un escalar.
    """)

    st.subheader("Configuración")
    col1, col2 = st.columns([1, 1])

    with col1:
        funcion_seleccionada = st.selectbox(
            "Selecciona una función multivariada:",
            list(funciones_multivariadas.keys()),
            key="unid_func_select"
        )
        st.latex(funciones_multivariadas[funcion_seleccionada]["latex"])
        st.info(f"Dominio: {funciones_multivariadas[funcion_seleccionada]['dominio']}")

    with col2:
        st.markdown("#### Punto de Partida $x^{(t)}$")
        intervalos_func = funciones_multivariadas[funcion_seleccionada]["intervalos"]
        dim = len(intervalos_func)
        valores_x_t = []
        for i in range(dim):
            min_val = float(intervalos_func[i][0]) if intervalos_func[i][0] is not None else -10.0
            max_val = float(intervalos_func[i][1]) if intervalos_func[i][1] is not None else 10.0
            default_value = (min_val + max_val) / 2.0
            if not (min_val <= default_value <= max_val):
                default_value = min_val
            x_val = st.number_input(
                f"$x_{i+1}$ para $x^{{(t)}}$:",
                min_value=min_val, max_value=max_val, value=default_value, step=0.1, key=f"x_t_{i}_unid"
            )
            valores_x_t.append(x_val)
        x_t = np.array(valores_x_t)

        st.markdown("#### Dirección de Búsqueda $s^{(t)}$")
        valores_s_t = []
        for i in range(dim):
            s_val = st.number_input(
                f"$s_{i+1}$ para $s^{{(t)}}$:",
                min_value=-10.0, max_value=10.0, value=1.0, step=0.1, key=f"s_t_{i}_unid"
            )
            valores_s_t.append(s_val)
        s_t = np.array(valores_s_t)

    col3, col4 = st.columns(2)
    with col3:
        alpha_inicio = st.number_input("$\\alpha_{0}$ inicial:", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
    with col4:
        paso_alpha = st.number_input("Paso para $\\alpha$:", min_value=0.001, max_value=10.0, value=0.1, step=0.01)
    max_iteraciones = st.slider("Máximo de iteraciones:", min_value=10, max_value=500, value=100)

    if st.button("Ejecutar Búsqueda"):
        st.markdown("---")
        funcion_a_optimizar = funciones_multivariadas[funcion_seleccionada]["funcion"]
        x_optimo, alpha_optimo, historial, iter_info_pos, iter_info_neg = busqueda_unidireccional_core(
            funcion_a_optimizar, x_t, s_t, alpha_inicio, paso_alpha, max_iteraciones
        )

        if x_optimo is not None:
            st.subheader("Resultados")
            with st.container():
                st.success(f"**Alpha Óptimo**: $\\alpha^* = {alpha_optimo:.6f}$")
                st.success(f"**Punto Óptimo**: $x(\\alpha^*) = {np.round(x_optimo, 4)}$")
                st.success(f"**Valor Mínimo**: $f(x(\\alpha^*)) = {funcion_a_optimizar(x_optimo):.6f}$")

            st.subheader("Visualización")
            col_plot1, col_plot2 = st.columns([1, 1] if dim == 2 else [1, 0])

            with col_plot1:
                # Plot: Objective function vs. alpha
                alphas = [h[0] for h in historial]
                objetivos = [h[1] for h in historial]
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                ax1.plot(alphas, objetivos, marker='o', linestyle='-', markersize=4, label='Valor Objetivo')
                ax1.axvline(alpha_optimo, color='r', linestyle='--', label=f'Alpha Óptimo: {alpha_optimo:.2f}')
                ax1.set_title('Valor Objetivo vs. $\\alpha$')
                ax1.set_xlabel('$\\alpha$')
                ax1.set_ylabel('f(x)')
                ax1.grid(True)
                ax1.legend()
                st.pyplot(fig1)

            if dim == 2:
                with col_plot2:
                    # Plot: 2D search path
                    fig2, ax2 = plt.subplots(figsize=(6, 6))
                    puntos_linea_x = [x_t[0] + alpha * s_t[0] for alpha in alphas]
                    puntos_linea_y = [x_t[1] + alpha * s_t[1] for alpha in alphas]
                    ax2.plot(puntos_linea_x, puntos_linea_y, marker='x', linestyle='-', color='b', label='Ruta de Búsqueda')
                    ax2.plot(x_t[0], x_t[1], 'go', markersize=8, label='Punto Inicial')
                    ax2.plot(x_optimo[0], x_optimo[1], 'ro', markersize=8, label='Punto Óptimo')
                    ax2.quiver(x_t[0], x_t[1], s_t[0], s_t[1], angles='xy', scale_units='xy', scale=1, color='purple', width=0.005, label='Dirección $s^{(t)}$')

                    # Contours
                    all_x_coords = puntos_linea_x + [x_t[0], x_optimo[0]]
                    all_y_coords = puntos_linea_y + [x_t[1], x_optimo[1]]
                    func_info = funciones_multivariadas[funcion_seleccionada]
                    if "minimo_coords" in func_info and func_info["minimo_coords"] is not None:
                        all_x_coords.append(func_info["minimo_coords"][0])
                        all_y_coords.append(func_info["minimo_coords"][1])
                    x_min_plot = min(all_x_coords) - 1
                    x_max_plot = max(all_x_coords) + 1
                    y_min_plot = min(all_y_coords) - 1
                    y_max_plot = max(all_y_coords) + 1
                    x_grid = np.linspace(x_min_plot, x_max_plot, 100)
                    y_grid = np.linspace(y_min_plot, y_max_plot, 100)
                    X, Y = np.meshgrid(x_grid, y_grid)
                    Z = funcion_a_optimizar(np.array([X, Y]))
                    valid_Z = Z[~np.isnan(Z) & (Z > 0)]
                    levels = np.logspace(np.log10(valid_Z.min() + 1e-6), np.log10(valid_Z.max() + 1e-6), 10) if valid_Z.size > 0 else np.linspace(Z.min(), Z.max(), 10)
                    ax2.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.5)
                    ax2.set_title('Ruta de Búsqueda en 2D')
                    ax2.set_xlabel('$x_1$')
                    ax2.set_ylabel('$x_2$')
                    ax2.axis('equal')
                    ax2.grid(True)
                    ax2.legend()
                    st.pyplot(fig2)
            else:
                st.info("Visualización 2D disponible solo para funciones de 2 variables.")

            st.subheader("Iteraciones")
            with st.expander("Explorando $\\alpha > 0$"):
                if iter_info_pos:
                    st.markdown("|$i$|$\\alpha$|$x$|$f(x)$|\n|---:|---:|:---|---:|")
                    for i, alpha, x, f_x in iter_info_pos:
                        st.markdown(f"|{i}|{alpha:.4f}|{x}|{f_x:.6f}|")
                    if len(iter_info_pos) == max_iteraciones:
                        st.warning("Límite de iteraciones alcanzado en dirección positiva.")
                    else:
                        st.warning(f"Valor objetivo dejó de disminuir en $\\alpha={iter_info_pos[-1][1]:.4f}$.")
                else:
                    st.info("No recopilado")

            with st.expander("Explorando $\\alpha < 0$"):
                if iter_info_neg:
                    st.markdown("|$i$|$\\alpha$|$x$|$f(x)$|\n|---:|---:|:---|---:|")
                    for i, alpha, x, f_x in iter_info_neg:
                        st.markdown(f"|{i}|{alpha:.4f}|{x}|{f_x:.6f}|")
                    if len(iter_info_neg) == max_iteraciones:
                        st.warning("Límite de iteraciones alcanzado en dirección negativa.")
                    else:
                        st.warning(f"Valor objetivo dejó de disminuir en $\\alpha={iter_info_neg[-1][1]:.4f}$.")
                else:
                    st.info("No recopilado")

            st.markdown("---")
            st.subheader("Contexto y Aplicaciones")
            with st.expander("Acerca de la Búsqueda Unidireccional"):
                st.markdown("""
                La **búsqueda unidireccional** (o búsqueda en línea) es un componente clave en muchos algoritmos de optimización, como el método del gradiente descendente o los métodos de descenso por coordenadas. Su objetivo es encontrar el valor óptimo de $\\alpha$ que minimiza una función objetivo a lo largo de una dirección específica $s^{(t)}$ desde un punto inicial $x^{(t)}$. Este proceso es esencial en optimización no lineal, donde se busca reducir una función multivariable paso a paso.

                ### ¿Cómo funciona?
                - **Punto inicial y dirección**: Se parte de un punto $x^{(t)}$ y una dirección $s^{(t)}$ (que puede ser, por ejemplo, el gradiente negativo en el caso del descenso de gradiente).
                - **Exploración de $\\alpha$**: Se evalúa la función objetivo en puntos $x(\\alpha) = x^{(t)} + \\alpha s^{(t)}$ para diferentes valores de $\\alpha$, tanto positivos como negativos.
                - **Criterio de parada**: La búsqueda se detiene cuando el valor de la función objetivo deja de disminuir, indicando que se ha encontrado un mínimo local a lo largo de la dirección.

                ### Aplicaciones
                - **Optimización Numérica**: Utilizada en algoritmos como el descenso de gradiente, métodos conjugados o métodos cuasi-Newton.
                - **Aprendizaje Automático**: Fundamental para optimizar funciones de pérdida en modelos como redes neuronales.
                - **Ingeniería y Física**: Usada para minimizar funciones de costo en diseño de sistemas, ajuste de modelos o simulaciones físicas.

                ### Limitaciones
                - La búsqueda unidireccional asume que la función es razonablemente suave a lo largo de la dirección de búsqueda.
                - El método presentado aquí usa un enfoque simple con pasos fijos ($\\alpha$), pero métodos más avanzados (como la búsqueda de intervalo dorado o interpolación cuadrática) pueden ser más eficientes.
                - La elección de la dirección $s^{(t)}$ es crítica; una mala dirección puede llevar a convergencia lenta o a mínimos locales no deseados.

                Este código proporciona una implementación básica y educativa de la búsqueda unidireccional, con visualizaciones que ayudan a entender cómo el algoritmo explora el espacio de búsqueda y encuentra el óptimo a lo largo de una línea.
                """)

        else:
            st.error("Error en la configuración de la búsqueda.")