import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def random_generation(x, delta=0.1, bounds=None):
    """Genera una nueva solución vecina agregando ruido aleatorio."""
    x_new = x + np.random.uniform(-delta, delta, size=np.shape(x))
    if bounds is not None:
        x_new = np.clip(x_new, bounds[0], bounds[1])
    return x_new

def termination_criterion(max_iter):
    """Crea una función de terminación basada en el número de iteraciones."""
    current_iter = 0
    def terminate():
        nonlocal current_iter
        current_iter += 1
        return current_iter > max_iter
    return terminate

def hill_climbing_algorithm(objective_func, x0, max_iter, delta, bounds):
    """
    Ejecuta el algoritmo de Hill Climbing.
    Tu lógica original, empaquetada para devolver el historial completo.
    """
    x_current = np.copy(x0)
    
    # Pre-evaluación para evitar cálculos repetidos
    f_current = objective_func(x_current)
    if f_current is None:
        st.error(f"El punto inicial {x0} no se puede evaluar. Intenta con otro.")
        return None, [], [], []

    history_f = [f_current]
    solution_path = [np.copy(x_current)]
    improvements = 0
    
    terminate = termination_criterion(max_iter)

    while not terminate():
        x_next = random_generation(x_current, delta, bounds)
        f_next = objective_func(x_next)
        
        # Si no se puede evaluar el vecino, se ignora
        if f_next is None:
            history_f.append(f_current)
            solution_path.append(np.copy(x_current))
            continue

        # Solo se acepta el movimiento si es una mejora estricta
        if f_next < f_current:
            x_current = np.copy(x_next)
            f_current = f_next
            improvements += 1
        
        history_f.append(f_current)
        solution_path.append(np.copy(x_current))
    
    return x_current, history_f, solution_path, improvements

# ==============================================================================
# INTERFAZ DE STREAMLIT
# ==============================================================================

def plot_hc_results(ax, func, bounds, history_f, solution_path):
    """
    Dibuja los resultados del Hill Climbing: mapa de contorno y convergencia.
    """
    # ---- Subplot 1: Mapa de Contorno y Ruta ----
    ax1 = ax[0]
    min_b, max_b = [b[0] for b in bounds], [b[1] for b in bounds]
    x_vals = np.linspace(min_b[0], max_b[0], 100)
    y_vals = np.linspace(min_b[1], max_b[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.array([func(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

    ax1.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
    
    path_points = np.array(solution_path)
    ax1.plot(path_points[:, 0], path_points[:, 1], 'r-', alpha=0.7, label='Ruta de Búsqueda')
    ax1.scatter(path_points[0, 0], path_points[0, 1], c='lime', s=100, marker='o', edgecolor='k', label='Inicio', zorder=5)
    ax1.scatter(path_points[-1, 0], path_points[-1, 1], c='gold', s=150, marker='*', edgecolor='k', label='Final (Óptimo Local)', zorder=5)
    
    ax1.set_title("Ruta en el Espacio de Soluciones")
    ax1.set_xlabel('x_0')
    ax1.set_ylabel('x_1')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ---- Subplot 2: Gráfica de Convergencia ----
    ax2 = ax[1]
    ax2.plot(history_f, 'b-', linewidth=2)
    ax2.set_title("Convergencia del Algoritmo")
    ax2.set_xlabel("Iteraciones")
    ax2.set_ylabel("Valor de la Función Objetivo")
    ax2.grid(True, linestyle='--', alpha=0.5)

def show_hill_climbing(FUNCIONES, evaluar_funcion):
    st.markdown("## ⛰️ Ascenso/Descenso de la Colina (Hill Climbing)")
    st.markdown("""
    Hill Climbing es un algoritmo de búsqueda local que intenta encontrar el mínimo de una función de forma iterativa. Su lógica es muy simple:
    
    1.  Comienza en un punto aleatorio.
    2.  Genera un "vecino" cercano.
    3.  Si el vecino es **mejor** (tiene un valor de función más bajo), se mueve a ese punto.
    4.  Si el vecino **no es mejor**, se queda en el punto actual.
    5.  Repite hasta que se cumple un criterio de parada (ej. número de iteraciones).
    
    La principal debilidad de este método es que puede quedar **atrapado en óptimos locales**, ya que nunca aceptará un movimiento que empeore la solución actual, aunque ese movimiento fuera necesario para salir de un "valle" y encontrar un valle más profundo.
    """)

    with st.expander("Configuración del Algoritmo y la Función", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            funcion_seleccionada = st.selectbox("🎯 Selecciona la función:", list(FUNCIONES.keys()), key="hc_func")
            info_funcion = FUNCIONES[funcion_seleccionada]
            
            st.markdown("**Parámetros de Ejecución:**")
            delta = st.slider("δ (Tamaño del vecindario/paso)", 0.01, 2.0, 0.1, 0.01)
            max_iter = st.slider("🔄 Máximo de iteraciones:", 100, 10000, 1000, 100)

        with col2:
            st.latex(info_funcion['latex'])
            st.markdown("**📍 Punto Inicial (x₀):**")
            bounds_list = info_funcion["intervalos"]
            
            x0_cols = st.columns(len(bounds_list))
            x0_vals = [c.number_input(f'x0_{i}', value=np.random.uniform(bounds_list[i][0], bounds_list[i][1]), key=f'hc_x0_{i}') for i, c in enumerate(x0_cols)]
            x0 = np.array(x0_vals)

    if st.button("🚀 Ejecutar Hill Climbing", key="run_hc"):
        func_objetivo = lambda x: evaluar_funcion(x, funcion_seleccionada)
        bounds_clip = ([b[0] for b in bounds_list], [b[1] for b in bounds_list])
        
        with st.spinner("Buscando la cima (o el valle)..."):
            results = hill_climbing_algorithm(
                func_objetivo, x0, max_iter, delta, bounds_clip
            )

        if results[0] is None:
            return # Detener si el algoritmo no pudo iniciar

        x_best, history_f, solution_path, improvements = results
        
        st.success("✅ Búsqueda completada!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Solución Encontrada", f"[{x_best[0]:.4f}, {x_best[1]:.4f}]")
        col2.metric("📊 Valor Óptimo (Local)", f"{history_f[-1]:.6f}")
        col3.metric("📈 Mejoras Realizadas", f"{improvements} / {max_iter}")

        st.markdown("### 📈 Visualización de Resultados")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        plot_hc_results(axes, func_objetivo, info_funcion["intervalos"], history_f, solution_path)
        st.pyplot(fig)
        
        st.markdown("### 📋 Historial de los últimos 10 pasos")
        last_steps = pd.DataFrame({
            'Iteración': range(max_iter - 9, max_iter + 1),
            'f(x)': history_f[-10:],
            'Posición': [f"[{p[0]:.3f}, {p[1]:.3f}]" for p in solution_path[-10:]]
        })
        st.dataframe(last_steps, use_container_width=True)

        # --- Contexto y Aplicaciones ---
        st.markdown("---")
        st.subheader("Contexto y Aplicaciones")
        with st.expander("Acerca del Algoritmo Hill Climbing"):
            st.markdown("""
            El algoritmo **Hill Climbing** (Ascenso/Descenso de la Colina) es un método de optimización basado en búsqueda local que genera soluciones aleatorias a partir de un punto inicial y solo acepta movimientos que mejoren el valor de la función objetivo. Este enfoque lo hace eficiente para encontrar óptimos locales, pero puede quedar atrapado en ellos.

            ### ¿Cómo funciona?
            - **Inicialización**: Comienza con un punto inicial \( \mathbf{x}_0 \) y evalúa la función objetivo \( f(\mathbf{x}_0) \).
            - **Generación de vecinos**: En cada iteración, genera un punto vecino \( \mathbf{x}_{k+1} \) a partir del punto actual \( \mathbf{x}_k \), típicamente sumando un ruido aleatorio controlado por un parámetro \( \delta \).
            - **Criterio de aceptación**: Solo se mueve a \( \mathbf{x}_{k+1} \) si \( f(\mathbf{x}_{k+1}) < f(\mathbf{x}_k) \), es decir, si el nuevo punto mejora el valor de la función objetivo.
            - **Criterio de parada**: Termina cuando se alcanza un número máximo de iteraciones (\( \text{max_iter} \)) o no se encuentran mejoras.

            ### Aplicaciones
            - **Optimización local**: Ideal para problemas donde se busca un óptimo local rápidamente, como en ajustes de parámetros en modelos simples.
            - **Problemas combinatorios**: Usado en problemas discretos, como la optimización de rutas o configuraciones en espacios de búsqueda finitos.
            - **Tareas de inteligencia artificial**: Aplicado en heurísticas para juegos, planificación o aprendizaje automático, donde se necesita una solución aceptable en poco tiempo.
            - **Prototipado rápido**: Útil como método inicial para explorar soluciones antes de aplicar algoritmos más complejos.

            ### Ventajas
            - **Simplicidad**: Fácil de implementar, requiere solo evaluaciones de la función objetivo y comparaciones.
            - **Eficiencia en óptimos locales**: Converge rápidamente a soluciones cercanas al punto inicial si son mejores.
            - **Bajo costo computacional**: No necesita cálculos de derivadas ni información compleja del problema.

            ### Limitaciones
            - **Atrapamiento en óptimos locales**: No acepta movimientos que empeoren la solución, lo que impide escapar de óptimos locales para alcanzar un óptimo global.
            - **Dependencia del punto inicial**: El resultado depende fuertemente de \( \mathbf{x}_0 \), y un mal punto inicial puede llevar a soluciones subóptimas.
            - **Sensible a \( \delta \)**: El tamaño del paso (\( \delta \)) afecta la exploración; un valor demasiado grande o pequeño puede reducir la efectividad.

            Este código implementa Hill Climbing de manera educativa, mostrando la trayectoria de búsqueda en un espacio 2D y la convergencia del valor de la función objetivo, facilitando la comprensión de su comportamiento de búsqueda local.
            """)