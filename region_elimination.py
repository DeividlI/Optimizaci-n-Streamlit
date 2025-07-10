import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import tempfile
import os
import io

def region_elimination_rules(f_x1, f_x2, a, x1, x2, b):
    if not (a <= x1 < x2 <= b):
        return None, None, f"Error: Los puntos deben estar ordenados tal que {a:.4f} <= x1 < x2 <= {b:.4f}. Se recibieron: x1={x1:.4f}, x2={x2:.4f}"

    if f_x1 > f_x2:
        message = f"Condición: f(x1) > f(x2) ({f_x1:.4f} > {f_x2:.4f}). El mínimo no está en ({a:.4f}, {x1:.4f})."
        new_a, new_b = x1, b
    elif f_x1 < f_x2:
        message = f"Condición: f(x1) < f(x2) ({f_x1:.4f} < {f_x2:.4f}). El mínimo no está en ({x2:.4f}, {b:.4f})."
        new_a, new_b = a, x2
    else:
        message = f"Condición: f(x1) = f(x2) ({f_x1:.4f} = {f_x2:.4f}). El mínimo no está en ({a:.4f}, {x1:.4f}) y ({x2:.4f}, {b:.4f})."
        new_a, new_b = x1, x2
    
    return new_a, new_b, message

def plot_region_elimination(func, a, b, x1, x2, new_a, new_b, f_x1, f_x2, title_suffix=""):
    x_plot_min = min(a, x1, x2, new_a or a) - (b - a) * 0.1
    x_plot_max = max(b, x1, x2, new_b or b) + (b - a) * 0.1
    x_vals = np.linspace(x_plot_min, x_plot_max, 500)
    y_vals = np.array([func(val) for val in x_vals])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_vals, y_vals, label='f(x)', color='blue')

    ax.plot(x1, f_x1, 'ro', markersize=8, label=f'x1 ({x1:.2f}, {f_x1:.2f})')
    ax.plot(x2, f_x2, 'go', markersize=8, label=f'x2 ({x2:.2f}, {f_x2:.2f})')
    
    y_lim = ax.get_ylim()

    ax.axvline(a, color='purple', linestyle='--', linewidth=1, label=f'a ({a:.2f})')
    ax.text(a, y_lim[1]*0.95, f'a={a:.2f}', color='purple', ha='center', va='top')

    ax.axvline(b, color='purple', linestyle='--', linewidth=1, label=f'b ({b:.2f})')
    ax.text(b, y_lim[1]*0.95, f'b={b:.2f}', color='purple', ha='center', va='top')

    ax.add_patch(patches.Rectangle((a, y_lim[0]), b - a, y_lim[1] - y_lim[0], 
                                   facecolor='lightseagreen', alpha=0.1, label='Intervalo Inicial'))

    if new_a is not None and new_b is not None:
        if new_a > a:
            ax.add_patch(patches.Rectangle((a, y_lim[0]), new_a - a, y_lim[1] - y_lim[0], 
                                           facecolor='red', alpha=0.3, label='Región Eliminada'))
        if new_b < b:
            ax.add_patch(patches.Rectangle((new_b, y_lim[0]), b - new_b, y_lim[1] - y_lim[0], 
                                           facecolor='red', alpha=0.3))
        
        ax.add_patch(patches.Rectangle((new_a, y_lim[0]), new_b - new_a, y_lim[1] - y_lim[0], 
                                       facecolor='darkgreen', alpha=0.2, label='Nuevo Intervalo'))
        ax.text((new_a + new_b)/2, y_lim[0]*1.05, f'Nuevo Intervalo\n[{new_a:.2f}, {new_b:.2f}]', 
                color='darkgreen', ha='center', va='bottom', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    ax.set_title(f'Visualización del Método de Eliminación de Regiones {title_suffix}')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)

    return fig

def plot_region_elimination_animation(func, a, b, x1, x2, f_x1, f_x2, new_a, new_b):
    """
    Crea una animación mostrando el proceso del método de eliminación de regiones
    """
    x_plot_min = min(a, x1, x2, new_a or a) - (b - a) * 0.1
    x_plot_max = max(b, x1, x2, new_b or b) + (b - a) * 0.1
    x_vals = np.linspace(x_plot_min, x_plot_max, 500)
    y_vals = np.array([func(val) for val in x_vals])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, y_vals, label='f(x)', color='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Animación del Método de Eliminación de Regiones')
    ax.grid(True, linestyle=':', alpha=0.7)

    y_lim = ax.get_ylim()

    # Inicializar elementos de la gráfica
    point_x1, = ax.plot([], [], 'ro', markersize=8, label=f'x1 ({x1:.2f}, {f_x1:.2f})')
    point_x2, = ax.plot([], [], 'go', markersize=8, label=f'x2 ({x2:.2f}, {f_x2:.2f})')
    line_a = ax.axvline(a, color='purple', linestyle='--', linewidth=1, label=f'a ({a:.2f})')
    text_a = ax.text(a, y_lim[1]*0.95, f'a={a:.2f}', color='purple', ha='center', va='top')
    line_b = ax.axvline(b, color='purple', linestyle='--', linewidth=1, label=f'b ({b:.2f})')
    text_b = ax.text(b, y_lim[1]*0.95, f'b={b:.2f}', color='purple', ha='center', va='top')
    initial_span = ax.add_patch(patches.Rectangle((a, y_lim[0]), b - a, y_lim[1] - y_lim[0], 
                                                 facecolor='lightseagreen', alpha=0.1, label='Intervalo Inicial'))
    eliminated_span = ax.add_patch(patches.Rectangle((a, y_lim[0]), 0, y_lim[1] - y_lim[0], 
                                                    facecolor='red', alpha=0.3, label='Región Eliminada'))
    new_span = ax.add_patch(patches.Rectangle((a, y_lim[0]), 0, y_lim[1] - y_lim[0], 
                                             facecolor='darkgreen', alpha=0.2, label='Nuevo Intervalo'))
    new_interval_text = ax.text((a+b)/2, y_lim[0]*1.05, '', color='darkgreen', ha='center', va='bottom', 
                               fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    ax.legend()

    def init():
        point_x1.set_data([], [])
        point_x2.set_data([], [])
        eliminated_span.set_width(0)
        new_span.set_width(0)
        new_interval_text.set_text('')
        return point_x1, point_x2, eliminated_span, new_span, new_interval_text

    def update(frame):
        if frame == 0:
            # Mostrar solo el intervalo inicial
            point_x1.set_data([], [])
            point_x2.set_data([], [])
            eliminated_span.set_width(0)
            new_span.set_width(0)
            new_interval_text.set_text('')
        elif frame == 1:
            # Mostrar x1
            point_x1.set_data([x1], [f_x1])
            point_x2.set_data([], [])
            eliminated_span.set_width(0)
            new_span.set_width(0)
            new_interval_text.set_text('')
        elif frame == 2:
            # Mostrar x1 y x2
            point_x1.set_data([x1], [f_x1])
            point_x2.set_data([x2], [f_x2])
            eliminated_span.set_width(0)
            new_span.set_width(0)
            new_interval_text.set_text('')
        elif frame == 3:
            # Mostrar región eliminada y nuevo intervalo
            point_x1.set_data([x1], [f_x1])
            point_x2.set_data([x2], [f_x2])
            if new_a > a:
                eliminated_span.set_xy((a, y_lim[0]))
                eliminated_span.set_width(new_a - a)
            elif new_b < b:
                eliminated_span.set_xy((new_b, y_lim[0]))
                eliminated_span.set_width(b - new_b)
            else:
                eliminated_span.set_width(0)
            new_span.set_xy((new_a, y_lim[0]))
            new_span.set_width(new_b - new_a)
            new_interval_text.set_text(f'Nuevo Intervalo\n[{new_a:.2f}, {new_b:.2f}]')
        return point_x1, point_x2, eliminated_span, new_span, new_interval_text

    # Crear la animación
    ani = animation.FuncAnimation(fig, update, frames=4, init_func=init, interval=1000)

    # Guardar la animación como GIF en un archivo temporal
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as temp_file:
        temp_file_path = temp_file.name
        ani.save(temp_file_path, writer='pillow', fps=1)

    # Leer el archivo temporal en un BytesIO para Streamlit
    with open(temp_file_path, 'rb') as file:
        output = io.BytesIO(file.read())

    # Eliminar el archivo temporal
    os.unlink(temp_file_path)

    plt.close(fig)
    output.seek(0)

    return output

def show_region_elimination(funciones_univariadas, evaluar_funcion_univariada):
    st.header("Método de Eliminación de Regiones")
    st.markdown("""
    Este método es fundamental en la optimización univariada para reducir el intervalo de búsqueda del mínimo de una función unimodal. Se basa en comparar los valores de la función en dos puntos internos del intervalo.
    """)

    st.subheader("Selecciona una Función")
    selected_function_name = st.selectbox(
        "Elige una función univariada",
        list(funciones_univariadas.keys()),
        key="re_func_selector"
    )

    func_info = funciones_univariadas[selected_function_name]
    
    if "intervalos" in func_info and func_info["intervalos"]:
        initial_a, initial_b, func_lambda = func_info["intervalos"][0]
        st.write(f"Función seleccionada: $${func_info['latex']}$$")
        st.write(f"Dominio de la función: `{func_info['dominio']}`")
    else:
        st.error("No se encontraron intervalos definidos para la función seleccionada.")
        return

    st.subheader("Parámetros del Intervalo y Puntos de Prueba")

    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("Límite inferior del intervalo (a)", value=float(initial_a), key="re_a_input")
    with col2:
        b = st.number_input("Límite superior del intervalo (b)", value=float(initial_b), key="re_b_input")

    st.markdown("---")
    st.subheader("Define los puntos de prueba $x_1$ y $x_2$")
    st.info("Asegúrate de que los puntos de prueba estén dentro del intervalo (a, b) y que $x_1 < x_2$.")

    col3, col4 = st.columns(2)
    with col3:
        suggested_x1 = a + (b - a) / 3
        if suggested_x1 >= b - 0.001: suggested_x1 = (a + b) / 2 - (b - a) * 0.05
        if suggested_x1 <= a + 0.001: suggested_x1 = (a + b) / 2 + (b - a) * 0.05
        x1 = st.number_input("Punto x1", value=float(suggested_x1), min_value=float(a), max_value=float(b), key="re_x1_input")
    with col4:
        suggested_x2 = b - (b - a) / 3
        if suggested_x2 <= a + 0.001 or suggested_x2 <= x1 + 0.001: suggested_x2 = (a + b) / 2 + (b - a) * 0.05
        if suggested_x2 >= b - 0.001: suggested_x2 = b - (b - a) * 0.1
        x2 = st.number_input("Punto x2", value=float(suggested_x2), min_value=float(a), max_value=float(b), key="re_x2_input")

    if x1 >= x2:
        st.warning("Advertencia: Se recomienda que el punto x1 sea menor que x2 para un comportamiento esperado del algoritmo.")

    if st.button("Aplicar Reglas de Eliminación de Regiones", key="re_run_button"):
        if not (a <= x1 < x2 <= b):
            st.error(f"Error: Los puntos deben estar ordenados tal que {a:.4f} <= x1 < x2 <= {b:.4f}. Se recibieron: x1={x1:.4f}, x2={x2:.4f}")
            return

        f_x1 = evaluar_funcion_univariada(x1, selected_function_name)
        f_x2 = evaluar_funcion_univariada(x2, selected_function_name)

        if f_x1 is None or f_x2 is None:
            st.error("No se pudieron evaluar las funciones en los puntos dados. Verifica el dominio.")
            return

        new_a, new_b, message = region_elimination_rules(f_x1, f_x2, a, x1, x2, b)

        st.subheader("Animación del Proceso")
        gif_output = plot_region_elimination_animation(func_lambda, a, b, x1, x2, f_x1, f_x2, new_a, new_b)
        st.image(gif_output, caption="Progreso del algoritmo por iteración", use_container_width=True)

        st.subheader("Resultados Finales de la Eliminación de Regiones")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Intervalo Original", f"[{a:.4f}, {b:.4f}]")
            st.metric("Longitud Original", f"{b-a:.4f}")
        with col2:
            st.metric("Nuevo Intervalo", f"[{new_a:.4f}, {new_b:.4f}]")
            st.metric("Longitud Nueva", f"{new_b-new_a:.4f}")

        st.markdown("### Detalles del Proceso")
        st.write(f"Valor de f(x1) = f({x1:.4f}) = {f_x1:.4f}")
        st.write(f"Valor de f(x2) = f({x2:.4f}) = {f_x2:.4f}")
        st.success(message)

        st.markdown("---")
        st.subheader("Explicación de las Reglas:")
        st.markdown("""
        1. **Si $f(x_1) > f(x_2)$:** El mínimo no está en $(a, x_1)$. Nuevo intervalo: $[x_1, b]$.
        2. **Si $f(x_1) < f(x_2)$:** El mínimo no está en $(x_2, b)$. Nuevo intervalo: $[a, x_2]$.
        3. **Si $f(x_1) = f(x_2)$:** El mínimo no está en $(a, x_1)$ ni en $(x_2, b)$. Nuevo intervalo: $[x_1, x_2]$.
        """)

        st.markdown("---")
        st.subheader("Contexto e Información Adicional")
        with st.expander("Más sobre el Método y la Implementación", expanded=False):
            st.markdown("""
            ### ¿Qué es el Método de Eliminación de Regiones?
            Es un algoritmo de optimización univariada que reduce el intervalo de búsqueda del mínimo de una función unimodal. Evalúa la función en dos puntos, `x1` y `x2`, y elimina regiones donde el mínimo no puede estar, basándose en las reglas mencionadas.

            ### Propósito de esta Implementación
            Esta aplicación interactiva, creada con **Streamlit**, permite:
            - Seleccionar una función (ej., `f(x) = x² + 54/x`).
            - Definir el intervalo `[a, b]` y puntos `x1`, `x2`.
            - Visualizar el proceso mediante un GIF animado que muestra:
              1. Intervalo inicial.
              2. Punto `x1`.
              3. Punto `x2`.
              4. Región eliminada y nuevo intervalo.
            - Ver resultados como valores de la función y el intervalo reducido.

            ### Componentes del Código
            - **`region_elimination_rules`**: Aplica las reglas de eliminación.
            - **`plot_region_elimination`**: Crea una gráfica estática del proceso.
            - **`plot_region_elimination_animation`**: Genera un GIF animado.
            - **`show_region_elimination`**: Define la interfaz y muestra resultados.

            ### Aplicaciones
            Útil en optimización de una variable, como minimizar costos en economía o mejorar diseños en ingeniería.

            ### Consejos
            - Asegúrate de que `a <= x1 < x2 <= b`.
            - Respeta el dominio de la función (ej., evita `x=0` en funciones con singularidades).
            - Esta herramienta es educativa, diseñada para enseñar optimización de forma visual.
            """)