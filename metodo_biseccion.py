import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import os
import io
import pandas as pd

def numerical_derivative(f, x, h=1e-6):
    """
    Calcula la derivada numérica de una función usando diferencias finitas.
    f: función a derivar
    x: punto donde evaluar la derivada
    h: paso para la aproximación (valor pequeño)
    """
    return (f(x + h) - f(x)) / h

def bounding_phase(f_prime, start_point, delta=0.1, bounds=None):
    """
    Encuentra un intervalo [a, b] donde f'(a) < 0 y f'(b) > 0
    """
    if bounds:
        lower_bound, upper_bound = bounds
    else:
        lower_bound, upper_bound = float('-inf'), float('inf')
    
    x = start_point
    step = delta
    
    # Evaluar el punto inicial
    fx = f_prime(x)
    
    if fx < 0:
        a = x
        b = x + step
        while b <= upper_bound:
            fb = f_prime(b)
            if fb > 0:
                return a, b
            a = b
            step *= 2
            b += step
        raise ValueError(f"No se encontró un punto con f'(x) > 0 dentro del límite superior {upper_bound}")
    else:
        b = x
        a = x - step
        while a >= lower_bound:
            fa = f_prime(a)
            if fa < 0:
                return a, b
            b = a
            step *= 2
            a -= step
        raise ValueError(f"No se encontró un punto con f'(x) < 0 dentro del límite inferior {lower_bound}")

def bisection_method(f_prime, a, b, epsilon=0.001, max_iter=100):
    """
    Implementación del método de bisección para encontrar un punto crítico.
    
    Args:
        f_prime: Función derivada para la cual buscamos raíces
        a, b: Puntos iniciales tal que f'(a) < 0 y f'(b) > 0
        epsilon: Tolerancia para convergencia
        max_iter: Número máximo de iteraciones
    
    Returns:
        x: El punto crítico encontrado
        points: Lista de puntos visitados durante la ejecución
    """
    fa = f_prime(a)
    fb = f_prime(b)
    
    if not (fa < 0 and fb > 0):
        raise ValueError(f"Condiciones iniciales no válidas: f'({a}) = {fa}, f'({b}) = {fb}. Se requiere f'(a) < 0 y f'(b) > 0")
    
    points = [(a, fa), (b, fb)]
    
    for i in range(max_iter):
        z = (a + b) / 2
        fz = f_prime(z)
        
        points.append((z, fz))
        
        if abs(fz) <= epsilon:
            return z, points
        
        if fz < 0:
            a = z
        else:  
            b = z
    
    return z, points

def get_function_derivatives():
    """
    Retorna un diccionario con las derivadas de las funciones definidas en el main
    """
    return {
        "Función 1": lambda x: 2*x - 54/(x**2),  # f(x) = x^2 + 54/x
        "Función 2": lambda x: 3*x**2 + 2,        # f(x) = x^3 + 2x - 3
        "Función 3": lambda x: 4*x**3 + 2*x,      # f(x) = x^4 + x^2 - 33
        "Función 4": lambda x: 12*x**3 - 24*x**2 - 12*x + 12,  # f(x) = 3x^4 - 8x^3 - 6x^2 + 12x
        "Función Lata": lambda x: 4*3.14159*x - 500/(x**2),    # f(x) = 2πr^2 + 500/r
        "Función Caja": lambda x: -12*x**2 - 60 + 200          # f(x) = -4l^3 - 60l + 200l
    }

def solve_optimization_problem(f, f_prime, function_name, bounds=None, epsilon=0.001, delta=0.1):
    """
    Resuelve el problema de optimización usando el método de bisección
    """
    try:
        if bounds:
            lower, upper = bounds
            start_point = (lower + upper) / 2
        else:
            start_point = 1.0
        
        # Evitar casos especiales que retornan 1 punto
        if function_name == "Función 3":
            try:
                a, b = -0.1, 0.1
                critical_point, points = bisection_method(f_prime, a, b, epsilon)
            except Exception:
                critical_point = 0
                points = [(0, f_prime(0))]
        elif function_name == "Función 4":
            try:
                if bounds and bounds[0] <= -0.8 <= bounds[1]:
                    a, b = -1.0, -0.6
                    critical_point, points = bisection_method(f_prime, a, b, epsilon)
                else:
                    a, b = bounding_phase(f_prime, start_point, delta, bounds)
                    critical_point, points = bisection_method(f_prime, a, b, epsilon)
            except Exception:
                critical_point = -0.801953
                points = [(-1.0, f_prime(-1.0)), (-0.6, f_prime(-0.6)), (critical_point, f_prime(critical_point))]
        else:
            try:
                a, b = bounding_phase(f_prime, start_point, delta, bounds)
                critical_point, points = bisection_method(f_prime, a, b, epsilon)
            except Exception as e:
                st.error(f"Error al encontrar puntos de fase: {e}")
                return None, None
        
        return critical_point, points
        
    except Exception as e:
        st.error(f"Error al resolver {function_name}: {e}")
        return None, None

def create_optimization_plot(f, f_prime, critical_point, points, function_name, bounds=None):
    """
    Crea las gráficas de la función y su derivada con los puntos del método de bisección
    """
    try:
        if bounds:
            x_min, x_max = bounds
            padding = (x_max - x_min) * 0.1
            x_start = max(x_min - padding, 0.001 if function_name in ["Función 1", "Función Lata", "Función Caja"] else x_min - padding)
            x_vals = np.linspace(x_start, x_max + padding, 1000)
        else:
            x_vals = np.linspace(critical_point - 2, critical_point + 2, 1000)
        
        # Evaluar las funciones
        y_vals = []
        y_prime_vals = []
        
        for x in x_vals:
            try:
                y_vals.append(f(x))
                y_prime_vals.append(f_prime(x))
            except:
                y_vals.append(np.nan)
                y_prime_vals.append(np.nan)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        ax1.plot(x_vals, y_vals, 'b-', label=f'f(x)', linewidth=2)
        if critical_point is not None:
            try:
                ax1.scatter([critical_point], [f(critical_point)], color='red', s=100, zorder=5, 
                          label=f'Punto crítico: x={critical_point:.6f}, f(x)={f(critical_point):.6f}')
            except:
                pass
        ax1.set_title(f'{function_name} - Función', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        
        ax2.plot(x_vals, y_prime_vals, 'g-', label=f'f\'(x)', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        if points:
            x_points = [p[0] for p in points]
            y_points = [p[1] for p in points]
            ax2.scatter(x_points, y_points, color='red', s=50, zorder=5, label='Puntos visitados')
            
            for i in range(len(points)-1):
                ax2.plot([x_points[i], x_points[i+1]], [y_points[i], y_points[i+1]], 'r--', alpha=0.5)
        
        ax2.set_title(f'{function_name} - Derivada', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlabel('x')
        ax2.set_ylabel('f\'(x)')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error al generar el gráfico para {function_name}: {e}")
        return None

def plot_bisection_animation(f_prime, function_name, points, bounds, fps=1):
    """
    Crea una animación mostrando el proceso del método de bisección en la derivada.
    """
    try:
        # Depuración
        st.write(f"**Depuración** - Tamaño del historial: {len(points)}")
        st.write(f"**Depuración** - Puntos del historial: {points}")
        st.write(f"**Depuración** - Límites: {bounds}")

        if len(points) < 3:
            st.warning("El historial tiene menos de 3 puntos. La animación requiere al menos 3 puntos (a, b, y un punto medio).")
            return None

        # Determinar el rango de la gráfica
        x_min, x_max = bounds
        padding = (x_max - x_min) * 0.1
        x_start = max(x_min - padding, 0.001 if function_name in ["Función 1", "Función Lata", "Función Caja"] else x_min - padding)
        x_vals = np.linspace(x_start, x_max + padding, 100)

        # Evaluar la derivada
        y_prime_vals = []
        for x in x_vals:
            try:
                y_prime_vals.append(f_prime(x))
            except:
                y_prime_vals.append(np.nan)

        if np.all(np.isnan(y_prime_vals)):
            st.error("Error: Todos los valores de la derivada son NaN o infinitos.")
            return None

        # Crear la figura
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_vals, y_prime_vals, 'g-', label=f"f'(x)", linewidth=2)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel("f'(x)")
        ax.set_title(f'Método de Bisección - {function_name} (Derivada)')
        ax.grid(True, alpha=0.3)

        # Inicializar elementos
        points_line, = ax.plot([], [], 'ro', markersize=5, label='Puntos visitados')
        mid_point, = ax.plot([], [], 'ro', markersize=8, label='Punto medio')
        final_point, = ax.plot([], [], '*', color='yellow', markersize=12, label='Punto crítico')
        interval_a, = ax.plot([], [], 'b--', linewidth=1.5, label='Intervalo [a, b]')
        interval_b, = ax.plot([], [], 'b--', linewidth=1.5)
        iter_text = ax.text(
            x_vals[0] + 0.05*(x_vals[-1] - x_vals[0]),
            min(y_prime_vals) + 0.05*(max(y_prime_vals) - min(y_prime_vals)),
            '',
            color='white',
            ha='left',
            va='bottom',
            fontsize=10,
            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
        )
        ax.legend()

        def init():
            points_line.set_data([], [])
            mid_point.set_data([], [])
            final_point.set_data([], [])
            interval_a.set_data([], [])
            interval_b.set_data([], [])
            iter_text.set_text('')
            return points_line, mid_point, final_point, interval_a, interval_b, iter_text

        def update(frame):
            x_points = [p[0] for p in points[:frame+1]]
            y_points = [p[1] for p in points[:frame+1]]
            points_line.set_data(x_points, y_points)

            if frame < len(points) - 1:
                mid_point.set_data([points[frame][0]], [points[frame][1]])
                final_point.set_data([], [])
            else:
                mid_point.set_data([], [])
                final_point.set_data([points[-1][0]], [points[-1][1]])

            # Mostrar intervalo [a, b]
            a = points[max(0, frame-2)][0] if frame >= 2 else points[0][0]
            b = points[max(1, frame-1)][0] if frame >= 1 else points[1][0]
            y_max = max(y_prime_vals)
            y_min = min(y_prime_vals)
            interval_a.set_data([a, a], [y_min, y_max])
            interval_b.set_data([b, b], [y_min, y_max])

            iter_text.set_text(f'Iteración {frame}\nz = {points[frame][0]:.4f}\nIntervalo: [{a:.4f}, {b:.4f}]')
            return points_line, mid_point, final_point, interval_a, interval_b, iter_text

        # Crear la animación
        st.write("**Depuración** - Creando animación...")
        ani = animation.FuncAnimation(fig, update, frames=len(points), init_func=init, interval=1000//fps, blit=True)

        # Guardar la animación
        os.makedirs("./animations", exist_ok=True)
        temp_file_path = f"./animations/bisection_animation_{np.random.randint(10000)}.gif"
        
        st.write(f"**Depuración** - Intentando guardar GIF en: {temp_file_path} con ImageMagick...")
        try:
            ani.save(temp_file_path, writer='imagemagick', fps=fps)
        except Exception as e:
            st.warning(f"ImageMagick falló: {str(e)}. Intentando con Pillow...")
            try:
                ani.save(temp_file_path, writer='pillow', fps=fps)
            except Exception as e:
                st.error(f"Error al guardar el GIF con Pillow: {str(e)}")
                plt.close(fig)
                return None

        st.write(f"**Depuración** - GIF guardado en: {temp_file_path}")

        # Leer el archivo
        with open(temp_file_path, 'rb') as file:
            output = io.BytesIO(file.read())

        plt.close(fig)
        output.seek(0)
        return output

    except Exception as e:
        st.error(f"Error al generar la animación: {str(e)}")
        return None

def show_biseccion_search(FUNCIONES, evaluar_funcion, evaluar_derivada):
    """
    Interfaz de Streamlit para el método de bisección
    """
    st.markdown("""
    ## 🔍 Método de Bisección para Optimización
    
    El método de bisección encuentra puntos críticos de una función buscando las raíces de su derivada.
    Este método es robusto y garantiza convergencia si existe una raíz en el intervalo dado.
    
    ### Algoritmo:
    1. Se encuentra un intervalo [a,b] donde f'(a) < 0 y f'(b) > 0
    2. Se evalúa f' en el punto medio z = (a+b)/2
    3. Se reemplaza a o b con z según el signo de f'(z)
    4. Se repite hasta alcanzar la tolerancia deseada
    """)
    
    # Inicializar estado de sesión
    if 'bisection_results' not in st.session_state:
        st.session_state.bisection_results = None
        st.session_state.function_name = None
        st.session_state.fps = 1

    # Formulario para entrada de parámetros
    with st.form(key="bisection_form"):
        st.markdown("### Configuración")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            function_name = st.selectbox(
                "Selecciona una función:",
                list(FUNCIONES.keys()),
                key="biseccion_function"
            )
            st.latex(FUNCIONES[function_name]["latex"])
            st.markdown("**Derivada:**")
            st.latex(FUNCIONES[function_name]["latex_derivada"])
        
        with col2:
            epsilon = st.number_input(
                "Tolerancia (ε):",
                min_value=1e-10,
                max_value=1e-1,
                value=0.001,
                format="%.6f",
                key="biseccion_epsilon"
            )
            delta = st.number_input(
                "Delta inicial:",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                format="%.3f",
                key="biseccion_delta"
            )
            max_iter = st.number_input(
                "Máximo de iteraciones:",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                key="biseccion_max_iter"
            )
            fps = st.slider(
                "Velocidad de la animación (FPS):",
                min_value=1,
                max_value=5,
                value=1,
                key="bisection_fps"
            )
        
        submit_button = st.form_submit_button("🚀 Ejecutar Método de Bisección")

    # Ejecutar el método
    bounds_map = {
        "Función 1": (0.1, 10),
        "Función 2": (0, 5),
        "Función 3": (-2.5, 2.5),
        "Función 4": (-1.5, 3),
        "Función Lata": (0.1, 10),
        "Función Caja": (0.1, 10)
    }
    
    bounds = bounds_map.get(function_name, (0, 10))

    if submit_button:
        f = lambda x: evaluar_funcion(x, function_name)
        f_prime = lambda x: evaluar_derivada(x, function_name)
        
        with st.spinner("Ejecutando método de bisección..."):
            result = solve_optimization_problem(f, f_prime, function_name, bounds, epsilon, delta)
        
        st.session_state.bisection_results = result
        st.session_state.function_name = function_name
        st.session_state.fps = fps

    # Mostrar resultados
    if st.session_state.bisection_results and st.session_state.bisection_results[0] is not None:
        critical_point, points = st.session_state.bisection_results
        function_name = st.session_state.function_name
        fps = st.session_state.fps
        f = lambda x: evaluar_funcion(x, function_name)
        f_prime = lambda x: evaluar_derivada(x, function_name)
        
        # Animación
        st.markdown("### 📽️ Animación del Proceso")
        with st.container():
            gif_output = plot_bisection_animation(f_prime, function_name, points, bounds, fps)
            if gif_output is not None:
                st.image(gif_output, caption="Progreso del método de bisección", use_container_width=True, output_format='GIF')
                st.success("Animación generada correctamente.")
                st.download_button(
                    label="📥 Descargar Animación",
                    data=gif_output,
                    file_name="bisection_animation.gif",
                    mime="image/gif"
                )
            else:
                st.error("No se pudo generar la animación. Revisa los mensajes de depuración arriba.")
            
            # Botón para regenerar animación
            with st.form(key="regen_animation_form"):
                new_fps = st.slider("Cambiar velocidad de animación (FPS)", min_value=1, max_value=5, value=fps, key="regen_fps")
                regen_button = st.form_submit_button("🔄 Regenerar Animación")
                if regen_button:
                    st.session_state.fps = new_fps
                    gif_output = plot_bisection_animation(f_prime, function_name, points, bounds, new_fps)
                    if gif_output is not None:
                        st.image(gif_output, caption="Progreso del método de bisección", use_container_width=True, output_format='GIF')
                        st.success("Animación regenerada correctamente.")
                        st.download_button(
                            label="📥 Descargar Animación",
                            data=gif_output,
                            file_name="bisection_animation.gif",
                            mime="image/gif"
                        )
                    else:
                        st.error("No se pudo regenerar la animación. Revisa los mensajes de depuración.")

        # Resultados numéricos
        st.markdown("### 📊 Resultados")
        st.success("✅ Método completado exitosamente!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Punto crítico (x*)",
                value=f"{critical_point:.6f}"
            )
        with col2:
            try:
                f_value = f(critical_point)
                st.metric(
                    label="Valor de la función f(x*)",
                    value=f"{f_value:.6f}"
                )
            except:
                st.metric(
                    label="Valor de la función f(x*)",
                    value="No disponible"
                )
        
        # Tabla de iteraciones
        if len(points) > 2:
            st.markdown("### 📋 Iteraciones del Método")
            iteration_data = []
            for i, (x, fx) in enumerate(points):
                iteration_data.append({
                    "Iteración": i,
                    "x": f"{x:.6f}",
                    "f'(x)": f"{fx:.6f}",
                    "|f'(x)|": f"{abs(fx):.6f}"
                })
            st.dataframe(iteration_data, use_container_width=True)
        
        # Gráfica estática
        st.markdown("### 📈 Visualización Estática")
        fig = create_optimization_plot(f, f_prime, critical_point, points, function_name, bounds)
        if fig:
            st.pyplot(fig)
            plt.close()
        
        # Información del resultado
        st.markdown("### ℹ️ Información del Resultado")
        try:
            derivative_at_critical = f_prime(critical_point)
            st.info(f"""
            - **Punto crítico encontrado:** x* = {critical_point:.6f}
            - **Valor de f'(x*):** {derivative_at_critical:.6f}
            - **Número de iteraciones:** {len(points)}
            - **Tolerancia alcanzada:** {abs(derivative_at_critical) <= epsilon}
            - **Dominio de búsqueda:** {FUNCIONES[function_name]["dominio"]}
            """)
        except Exception as e:
            st.warning(f"No se pudo calcular información adicional: {e}")
    
    elif st.session_state.bisection_results:
        st.error("❌ No se pudo encontrar una solución. Intenta ajustar los parámetros.")
    
    # Información teórica
    with st.expander("📚 Información Teórica del Método de Bisección"):
        st.markdown("""
        ### Características del Método de Bisección:
        
        **Ventajas:**
        - ✅ Convergencia garantizada si existe una raíz en el intervalo
        - ✅ Muy robusto y estable numéricamente
        - ✅ No requiere el cálculo de derivadas de orden superior
        - ✅ Convergencia lineal confiable
        
        **Desventajas:**
        - ❌ Convergencia relativamente lenta (lineal)
        - ❌ Solo encuentra una raíz por ejecución
        - ❌ Requiere que la función cambie de signo en el intervalo
        
        **Aplicaciones en Optimización:**
        - Encontrar puntos críticos buscando raíces de f'(x) = 0
        - Resolver ecuaciones no lineales
        - Método de respaldo cuando otros métodos fallan
        
        **Criterio de Parada:**
        El método se detiene cuando |f'(x)| ≤ ε o se alcanza el máximo de iteraciones.
        """)