# simulated_annealing.py
import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting
import pandas as pd

def tweak(X, domain):
    """
    Genera una solución vecina 'U' a partir de 'X'.
    Añade un pequeño valor aleatorio a cada componente de X,
    asegurando que el resultado esté dentro del dominio.
    """
    # El tamaño del paso es un 5% del rango del dominio para cada variable
    step_size = [(dom[1] - dom[0]) * 0.05 for dom in domain]
    # Genera el nuevo vector U
    U = [x_i + random.uniform(-s, s) for x_i, s in zip(X, step_size)]
    # Asegura que U esté dentro del dominio (clipping)
    # CORRECCIÓN: Usar 'domain' aquí en lugar de 'step_size'
    U_clipped = [max(min(u_i, dom[1]), dom[0]) for u_i, dom in zip(U, domain)]
    return np.array(U_clipped)

# --- Versión adaptada del algoritmo para guardar el historial ---
def simulated_annealing_algorithm(f, initial_solution, domain, w, initial_temp, alpha, stop_iterations):
    """
    Implementación del Recocido Simulado que guarda el historial para visualización.
    La lógica de decisión es idéntica a la tuya.
    """
    X = np.array(initial_solution)
    Best = np.array(initial_solution)
    T = float(initial_temp)

    # Historial para las gráficas
    history_best_f = [f(Best)]
    history_current_f = [f(X)]
    history_temp = [T]
    history_best_x = [Best.copy()] # Store the actual best X found
    history_current_x = [X.copy()] # Store the actual current X

    for i in range(stop_iterations):
        for _ in range(w):
            U = tweak(X, domain)
            f_U = f(U)
            f_X = f(X)

            if f_U < f(Best):
                Best = np.copy(U)
                X = np.copy(U)
            else:
                delta = f_U - f_X
                # Aceptar una peor solución con cierta probabilidad
                if T > 1e-10 and np.exp(-delta / T) >= random.uniform(0.0, 1.0):
                    X = np.copy(U)
        
        # Guardar estado al final de cada ciclo de enfriamiento
        history_best_f.append(f(Best))
        history_current_f.append(f(X))
        history_best_x.append(Best.copy())
        history_current_x.append(X.copy())

        # Enfriamiento
        T = alpha * T
        history_temp.append(T)

    return Best, f(Best), history_best_f, history_current_f, history_temp, history_best_x, history_current_x

def plot_sa_results(func, domain, history_best_f, history_current_f, history_temp, history_best_x, history_current_x):
    """
    Dibuja los resultados del Recocido Simulado, incluyendo:
    2. Gráfica 3D de la superficie de la función con la trayectoria de la solución (para 2 variables).
    3. Gráfica 2D de contorno de la función con la trayectoria de la solución (para 2 variables).
    """
    
    # List to hold figures to be returned
    figures = []

    # --- Figure 1: Convergence and Temperature Plot ---
    # COMENTADO: Si no quieres que esta gráfica se imprima, simplemente comenta o elimina el siguiente bloque.
    # fig_convergence = plt.figure(figsize=(10, 6))
    # ax1 = fig_convergence.add_subplot(111)
    
    # color_best = 'cyan'
    # color_current = 'magenta'
    # color_temp = 'gold'

    # ax1.set_xlabel('Ciclos de Enfriamiento')
    # ax1.set_ylabel('Valor de la Función Objetivo', color='white')
    # ax1.plot(history_best_f, color=color_best, linestyle='-', linewidth=2, label='Mejor Solución f(Best)')
    # ax1.plot(history_current_f, color=color_current, linestyle='--', alpha=0.7, linewidth=1.5, label='Solución Actual f(X)')
    # ax1.tick_params(axis='y', labelcolor='white')
    # ax1.legend(loc='upper left')
    # ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Temperatura (T)', color=color_temp)
    # ax2.plot(history_temp, color=color_temp, linestyle=':', linewidth=2, label='Temperatura')
    # ax2.tick_params(axis='y', labelcolor=color_temp)
    # ax2.legend(loc='upper right')
    
    # fig_convergence.tight_layout()
    # figures.append(("Convergencia y Temperatura", fig_convergence))


    # --- Figures for 2-variable functions (3D Surface and 2D Contour) ---
    if len(domain) == 2:
        x_min, x_max = domain[0]
        y_min, y_max = domain[1]

        x_grid = np.linspace(x_min, x_max, 50)
        y_grid = np.linspace(y_min, y_max, 50)
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        Z_mesh = np.array([[func(np.array([xi, yi])) for yi in y_grid] for xi in x_grid])

        best_x_coords = np.array([p[0] for p in history_best_x])
        best_y_coords = np.array([p[1] for p in history_best_x])
        best_z_coords = np.array([func(p) for p in history_best_x]) # Get Z values along the path

        current_x_coords = np.array([p[0] for p in history_current_x])
        current_y_coords = np.array([p[1] for p in history_current_x])
        current_z_coords = np.array([func(p) for p in history_current_x]) # Get Z values along the path


        # Figure 2: 3D Surface Plot
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='viridis', alpha=0.8, antialiased=True)
        
        ax_3d.plot(best_x_coords, best_y_coords, best_z_coords, 'o-', color='red', markersize=4, linewidth=2, label='Mejor Solución (X,Y,Z)')
        ax_3d.scatter(best_x_coords[0], best_y_coords[0], best_z_coords[0], marker='X', color='red', s=100, label='Inicio') # Start point
        
        # Optional: uncomment to show current path in 3D (can make plot busy)
        # ax_3d.plot(current_x_coords, current_y_coords, current_z_coords, 'o--', color='blue', markersize=2, linewidth=1, alpha=0.7, label='Solución Actual (X,Y,Z)')

        ax_3d.set_title('Superficie de la Función y Trayectoria (3D)')
        ax_3d.set_xlabel('X1')
        ax_3d.set_ylabel('X2')
        ax_3d.set_zlabel('f(X)')
        ax_3d.legend()
        ax_3d.view_init(elev=30, azim=45) # Set a good initial view angle
        fig_3d.tight_layout()
        figures.append(("Superficie y Trayectoria (3D)", fig_3d))

        # Figure 3: 2D Contour Plot
        fig_2d_contour = plt.figure(figsize=(8, 7))
        ax_2d_contour = fig_2d_contour.add_subplot(111)
        ax_2d_contour.contourf(X_mesh, Y_mesh, Z_mesh, levels=50, cmap='viridis', alpha=0.8)
        ax_2d_contour.contour(X_mesh, Y_mesh, Z_mesh, levels=50, colors='grey', linewidths=0.5, alpha=0.5)

        ax_2d_contour.plot(best_x_coords, best_y_coords, 'o-', color='red', markersize=3, linewidth=1.5, label='Mejor Solución (X,Y)')
        ax_2d_contour.plot(best_x_coords[0], best_y_coords[0], 'X', color='red', markersize=8, label='Inicio') # Start point

        ax_2d_contour.plot(current_x_coords, current_y_coords, 'o--', color='blue', markersize=2, linewidth=0.8, alpha=0.7, label='Solución Actual (X,Y)')

        ax_2d_contour.set_title('Mapa de Contorno de la Función y Trayectoria (2D)')
        ax_2d_contour.set_xlabel('X1')
        ax_2d_contour.set_ylabel('X2')
        ax_2d_contour.legend()
        ax_2d_contour.grid(True, linestyle=':', alpha=0.6)
        fig_2d_contour.tight_layout()
        figures.append(("Mapa de Contorno y Trayectoria (2D)", fig_2d_contour))

    else:
        # If not a 2D function, create a placeholder figure for the message
        fig_placeholder = plt.figure(figsize=(10, 4))
        ax_placeholder = fig_placeholder.add_subplot(111)
        ax_placeholder.text(0.5, 0.5, 'Las visualizaciones de superficie y contorno\n solo están disponibles para funciones con 2 variables.',
                            horizontalalignment='center', verticalalignment='center', transform=ax_placeholder.transAxes, fontsize=12, color='white')
        ax_placeholder.axis('off')
        fig_placeholder.tight_layout()
        figures.append(("Visualización de Función (Nota)", fig_placeholder))

    return figures

def show_simulated_annealing(FUNCIONES, evaluar_funcion):
    st.markdown("## 🔥 Recocido Simulado (Simulated Annealing)")
    st.markdown("""
    El Recocido Simulado es una metaheurística inspirada en el proceso de recocido en metalurgia. Es capaz de escapar de óptimos locales, a diferencia de Hill Climbing.
    1.  Comienza con una solución y una **temperatura `T`** alta.
    2.  Genera un vecino. Si es mejor, lo acepta.
    3.  Si el vecino es **peor**, lo puede aceptar con una probabilidad que depende de qué tan malo es y de la temperatura actual. Con `T` alta, es más probable aceptar malos movimientos; con `T` baja, es muy improbable.
    4.  Después de un número de iteraciones (`w`), la temperatura se reduce lentamente (`T = α * T`).
    5.  El proceso se repite hasta que el sistema está "congelado" (T es muy baja).
    Esta capacidad de aceptar movimientos peores le permite explorar más ampliamente el espacio de soluciones.
    """)

    with st.expander("Configuración del Algoritmo y la Función", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            funcion_seleccionada = st.selectbox("🎯 Selecciona la función:", list(FUNCIONES.keys()), key="sa_func")
            info_funcion = FUNCIONES[funcion_seleccionada]
            st.markdown("**Parámetros de Temperatura:**")
            initial_temp = st.number_input("🌡️ Temperatura Inicial (T₀)", value=1000.0, format="%.2f")
            alpha = st.slider("❄️ Factor de Enfriamiento (α)", 0.80, 0.99, 0.95, 0.01)
        with col2:
            st.latex(info_funcion['latex'])
            st.markdown("**Parámetros de Iteración:**")
            w = st.number_input("🔄 Iteraciones por Temperatura (w)", value=100)
            stop_iterations = st.number_input("⏳ Ciclos de Enfriamiento", value=500)

    if st.button("🚀 Ejecutar Recocido Simulado", key="run_sa"):
        func_objetivo = lambda x: evaluar_funcion(x, funcion_seleccionada)
        domain = info_funcion["intervalos"]

        if len(domain) == 0:
            st.error("Por favor, define un dominio para la función seleccionada.")
            return

        initial_solution = [random.uniform(dom[0], dom[1]) for dom in domain]

        with st.spinner("Enfriando el sistema... este proceso puede ser lento."):
            results = simulated_annealing_algorithm(
                func_objetivo, initial_solution, domain, w, initial_temp, alpha, stop_iterations
            )

        x_best, f_best, history_best_f, history_current_f, history_temp, history_best_x, history_current_x = results

        st.success("✅ Recocido Simulado completado!")

        col1, col2 = st.columns(2)
        if len(x_best) == 1:
            col1.metric("🎯 Mejor Solución Encontrada", f"[{x_best[0]:.4f}]")
        else:
            col1.metric("🎯 Mejor Solución Encontrada", f"[{', '.join([f'{val:.4f}' for val in x_best])}]")
        
        col2.metric("📊 Valor Óptimo Encontrado", f"{f_best:.6f}")

        st.markdown("### 📈 Visualización de Resultados")
        
        # Get all figures from the plotting function
        all_figures = plot_sa_results(func_objetivo, domain, history_best_f, history_current_f, history_temp, history_best_x, history_current_x)
        
        # Display each figure with a subheader
        for title, fig in all_figures:
            st.subheader(title)
            st.pyplot(fig)
            plt.close(fig) # Close the figure to free up memory

        st.markdown("### 📋 Tabla de Progreso (cada 10 ciclos)")
        step = 10
        sampled_indices = range(0, len(history_best_f), step)
        
        df_data = {
            'Ciclo de Enfriamiento': sampled_indices,
            'f(Best)': [history_best_f[i] for i in sampled_indices],
            'f(Current)': [history_current_f[i] for i in sampled_indices],
            'Temperatura': [history_temp[i] for i in sampled_indices]
        }
        
        if len(history_best_x) > 0:
            num_vars = len(history_best_x[0])
            for i in range(num_vars):
                df_data[f'Best_X{i+1}'] = [history_best_x[idx][i] for idx in sampled_indices]
                df_data[f'Current_X{i+1}'] = [history_current_x[idx][i] for idx in sampled_indices]

        st.dataframe(pd.DataFrame(df_data), use_container_width=True)

    # Nuevo apartado con información general sobre el Recocido Simulado
    st.markdown("---") # Separador visual
    st.markdown("### ℹ️ Información General sobre el Recocido Simulado")
    st.markdown("""
    El Recocido Simulado es una metaheurística de optimización global inspirada en el proceso de recocido en metalurgia, donde un material se calienta y luego se enfría lentamente para aumentar el tamaño de sus cristales y reducir defectos. En el contexto de la optimización:

    * **Inspiración:** Imita el proceso de enfriamiento lento de un material, donde a altas temperaturas las partículas tienen mucha energía y pueden moverse libremente, explorando el espacio. A bajas temperaturas, la movilidad disminuye, permitiendo que se asienten en un estado de baja energía (óptimo).
    * **Escape de Óptimos Locales:** A diferencia de algoritmos de búsqueda local como Hill Climbing, el Recocido Simulado tiene la capacidad de aceptar "malos" movimientos (soluciones peores) con una cierta probabilidad. Esta probabilidad disminuye a medida que la "temperatura" baja, lo que le permite salir de óptimos locales y explorar más ampliamente el espacio de soluciones para encontrar un óptimo global.
    * **Parámetros Clave:**
        * **Temperatura Inicial (T₀):** Una temperatura alta al inicio permite una exploración más aleatoria y una mayor probabilidad de aceptar soluciones peores.
        * **Factor de Enfriamiento (α):** Determina cómo la temperatura disminuye en cada ciclo (generalmente un valor entre 0 y 1, cercano a 1). Un enfriamiento lento permite una mejor exploración.
        * **Iteraciones por Temperatura (w):** El número de movimientos o evaluaciones de vecinos que se realizan por cada nivel de temperatura antes de enfriar.
    * **Proceso Iterativo:** El algoritmo comienza con una solución aleatoria y una temperatura alta. En cada ciclo, genera soluciones vecinas, las evalúa y las acepta basándose en la mejora o en una probabilidad (si es peor) que depende de la temperatura. La temperatura se reduce gradualmente hasta alcanzar un valor muy bajo, momento en el cual el algoritmo se "congela" y se converge a la mejor solución encontrada.
    """)