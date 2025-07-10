import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==============================================================================
# LÓGICA MATEMÁTICA DEL ALGORITMO
# ==============================================================================

def random_generation(x, delta=0.1, bounds=None):
    """
    Genera una nueva solución agregando ruido aleatorio.
    Funciona correctamente tanto para escalares como para vectores.
    """
    x_new = x + np.random.uniform(-delta, delta, size=np.shape(x))
    
    if bounds is not None:
        x_new = np.clip(x_new, bounds[0], bounds[1])
    
    return x_new

def random_walk_algorithm(objective_func, x0, max_iter, delta=0.1, bounds=None):
    """
    Algoritmo de caminata aleatoria adaptado y robustecido.
    """
    x_best = np.copy(x0)
    x_current = np.copy(x0)
    
    f_best = objective_func(x_best)
    if f_best is None:
        st.error(f"El punto inicial {x0} está fuera del dominio o causa un error. No se puede iniciar el algoritmo.")
        return None, [], [], [], []

    f_current = f_best
    
    history_best = [f_best]
    history_current = [f_current]
    solution_path = [np.copy(x_current)]
    iterations_data = []
    
    for iteration in range(max_iter):
        x_next = random_generation(x_current, delta, bounds)
        f_next = objective_func(x_next)
        
        if f_next is None:
            history_best.append(f_best)
            history_current.append(f_current)
            solution_path.append(np.copy(x_current))
            continue

        if f_next < f_best:
            f_best = f_next
            x_best = np.copy(x_next)
        
        x_current = np.copy(x_next)
        f_current = f_next
        
        history_best.append(f_best)
        history_current.append(f_current)
        solution_path.append(np.copy(x_current))
        
        iterations_data.append({
            'Iteración': iteration + 1,
            'x_actual': f"[{', '.join([f'{xi:.3f}' for xi in x_current])}]",
            'f(x_actual)': f_current,
            'x_mejor': f"[{', '.join([f'{xi:.3f}' for xi in x_best])}]",
            'f(x_mejor)': f_best
        })
    
    return x_best, history_best, history_current, solution_path, iterations_data

# ==============================================================================
# INTERFAZ DE STREAMLIT
# ==============================================================================

def show_random_walk(FUNCIONES, evaluar_funcion):
    """Función principal para mostrar el método Random Walk en Streamlit"""
    
    st.markdown("## 🎲 Random Walk (Caminata Aleatoria)")
    st.markdown("""
    El algoritmo de **Random Walk** es un método de optimización estocástico que explora el espacio de soluciones 
    mediante movimientos completamente aleatorios. A diferencia de otros métodos, acepta todos los movimientos 
    sin importar si mejoran o empeoran la función objetivo.
    """)
    st.markdown("### 📝 Pseudocódigo del Algoritmo")
    st.markdown("""
    ```
    Iniciar con un punto x_0 y evaluar f(x_0)
    Establecer x_best = x_0, f_best = f(x_0)
    Para k = 1 hasta max_iter:
        Generar x_next = x_current + ruido aleatorio
        Evaluar f_next = f(x_next)
        Si f_next < f_best:
            Actualizar x_best = x_next, f_best = f_next
        Establecer x_current = x_next, f_current = f_next
    Retornar x_best, f_best
    ```
    """)
    st.markdown("### 🔍 Características del Random Walk")
    st.markdown("""
    - **Exploración aleatoria**: Genera movimientos aleatorios dentro de un rango definido por delta.
    - **Aceptación incondicional**: Acepta todos los movimientos, lo que permite explorar ampliamente el espacio.
    - **Simplicidad**: Fácil de implementar, pero puede ser lento para converger al óptimo.
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        funcion_seleccionada = st.selectbox(
            "🎯 Selecciona la función objetivo:",
            list(FUNCIONES.keys()),
            key="random_walk_function"
        )
        
        max_iter = st.slider("🔄 Máximo de iteraciones:", 10, 1000, 100, 10, key="rw_iter")
        delta = st.slider("📏 Delta (tamaño del paso):", 0.01, 5.0, 0.5, 0.01, key="rw_delta")
    
    with col2:
        info_funcion = FUNCIONES[funcion_seleccionada]
        st.markdown(f"**Dominio:** `{info_funcion['dominio']}`")
        st.latex(info_funcion['latex'])

        st.markdown("**📍 Punto inicial (x₀):**")
        bounds_list = info_funcion["intervalos"]
        min_bounds = [b[0] for b in bounds_list]
        max_bounds = [b[1] for b in bounds_list]
        
        ui_cols = st.columns(len(bounds_list))
        x0_vals = []
        for i, col in enumerate(ui_cols):
            val = col.number_input(
                f'x{i}', 
                min_value=float(min_bounds[i]), 
                max_value=float(max_bounds[i]), 
                value=float(np.mean(bounds_list[i])),
                step=0.1,
                key=f'rw_x0_{i}'
            )
            x0_vals.append(val)
        
        x0 = np.array(x0_vals)
        bounds_clip = (min_bounds, max_bounds)

    if st.button("🚀 Ejecutar Random Walk", key="run_random_walk"):
        np.random.seed(42)
        func_objetivo = lambda x: evaluar_funcion(x, funcion_seleccionada)
        
        with st.spinner("Ejecutando Random Walk..."):
            results = random_walk_algorithm(func_objetivo, x0, max_iter, delta, bounds_clip)
        
        if results[0] is None:
            st.warning("El algoritmo no pudo ejecutarse. Revisa el punto inicial.")
            return

        x_best, history_best, history_current, solution_path, iterations_data = results
        
        st.success("✅ Random Walk completado!")

        def format_vector(vec):
            return f"[{', '.join([f'{v:.4f}' for v in vec])}]"

        col1, col2, col3 = st.columns(3)
        col1.metric("📍 Punto inicial", format_vector(x0))
        col2.metric("🎯 Mejor solución (x_best)", format_vector(x_best))
        col3.metric("📊 Valor óptimo f(x_best)", f"{history_best[-1]:.6f}")

        st.markdown("### 📈 Análisis de Convergencia y Exploración")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Análisis del Random Walk', fontsize=16, fontweight='bold')
        
        axes[0].plot(history_best, 'b-', linewidth=2, label='Mejor valor f(x_best)')
        axes[0].plot(history_current, 'r-', linewidth=1, label='Valor actual f(x_k)', alpha=0.6)
        axes[0].set_title('Convergencia del Algoritmo')
        axes[0].set_xlabel('Iteraciones')
        axes[0].set_ylabel('Valor de la Función')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        ax2 = axes[1]
        x_coords = np.array(solution_path)
        x_min, y_min = min_bounds
        x_max, y_max = max_bounds
        
        grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_z = np.zeros_like(grid_x)
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                grid_z[i, j] = func_objetivo(np.array([grid_x[i, j], grid_y[i, j]]))

        contour = ax2.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis', alpha=0.7)
        fig.colorbar(contour, ax=ax2, label='Valor de f(x,y)')
        
        ax2.plot(x_coords[:, 0], x_coords[:, 1], 'r-o', markersize=3, linewidth=1.5, label='Ruta de Búsqueda')
        ax2.plot(x_best[0], x_best[1], 'c*', markersize=15, label=f'Mejor Solución: {format_vector(x_best)}')
        ax2.set_title('Exploración del Espacio de Soluciones (2D)')
        ax2.set_xlabel('x₀')
        ax2.set_ylabel('x₁')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)
        
        st.markdown("### 📋 Tabla de Iteraciones")
        st.dataframe(pd.DataFrame(iterations_data), use_container_width=True)

        # --- Contexto y Aplicaciones ---
        st.markdown("---")
        st.subheader("Contexto y Aplicaciones")
        with st.expander("Acerca del Algoritmo Random Walk"):
            st.markdown("""
            El **algoritmo Random Walk** (Caminata Aleatoria), también conocido como *Drunkard's Walk*, es un método de optimización estocástico que explora el espacio de soluciones mediante movimientos completamente aleatorios. A diferencia de algoritmos que buscan mejorar la solución en cada iteración, Random Walk acepta todos los movimientos, lo que lo hace ideal para explorar ampliamente el espacio de búsqueda sin utilizar información adicional del problema, como gradientes.

            ### Aplicaciones
            - **Exploración inicial**: Útil para mapear el espacio de soluciones en problemas donde no se conoce la estructura de la función objetivo.
            - **Simulaciones Monte Carlo**: Empleado en simulaciones para modelar procesos aleatorios o explorar distribuciones.
            - **Base para otros algoritmos**: Sirve como punto de partida para métodos estocásticos más avanzados, como el recocido simulado.
            - **Problemas no diferenciables**: Aplicable a funciones discontinuas, discretas o ruidosas donde los métodos basados en gradientes no son viables.

            ### Ventajas
            - **Simplicidad extrema**: No requiere cálculos complejos ni derivadas, solo evaluaciones de la función objetivo.
            - **Flexibilidad**: Funciona en espacios de cualquier dimensión y con funciones no diferenciables.
            - **Exploración amplia**: La aceptación incondicional de movimientos permite cubrir grandes regiones del espacio de búsqueda.

            ### Limitaciones
            - **Convergencia lenta**: La aleatoriedad pura puede requerir muchas iteraciones para acercarse a un óptimo.
            - **No garantiza óptimos globales**: No tiene mecanismos para enfocarse en regiones prometedoras, lo que lo hace menos eficiente que métodos como Hill Climbing o Simulated Annealing.
            - **Dependencia de delta: El tamaño del paso (\(\delta\)) afecta significativamente la exploración; un valor inadecuado puede limitar la efectividad.

            Este código implementa el algoritmo Random Walk de manera educativa, mostrando visualmente la trayectoria de exploración en 2D y la evolución del valor de la función objetivo, lo que ayuda a comprender su comportamiento estocástico.
            """)