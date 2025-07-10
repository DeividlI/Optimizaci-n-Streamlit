import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import pandas as pd
import tempfile
import io
import os

def interval_halving(f, a, b, epsilon):
    """
    Método de optimización por intervalos por la mitad
    """
    L = b - a
    iteraciones = []
    iteration = 0
    
    while L > epsilon:
        x_m = (a + b) / 2
        f_xm = f(x_m)
        
        x1 = a + L / 4
        x2 = b - L / 4
        f_x1 = f(x1)
        f_x2 = f(x2)
        
        # Guardar información de la iteración
        iteraciones.append({
            'Iteración': iteration + 1,
            'a': a,
            'b': b,
            'L': L,
            'x_m': x_m,
            'x1': x1,
            'x2': x2,
            'f(x_m)': f_xm,
            'f(x1)': f_x1,
            'f(x2)': f_x2
        })
        
        if f_x1 < f_xm:
            b = x_m
            x_m = x1
        elif f_x2 < f_xm:
            a = x_m
            x_m = x2
        else:
            a = x1
            b = x2
        
        L = b - a
        iteration += 1
    
    return x_m, f(x_m), iteraciones

def plot_interval_halving_animation(f, a, b, iteraciones, epsilon):
    """
    Crea una animación mostrando el proceso del método de intervalos por la mitad
    """
    x_plot_min = min(a, min(iter_data['a'] for iter_data in iteraciones)) - (b - a) * 0.1
    x_plot_max = max(b, max(iter_data['b'] for iter_data in iteraciones)) + (b - a) * 0.1
    x_vals = np.linspace(x_plot_min, x_plot_max, 500)
    y_vals = np.array([f(x) if not np.isnan(f(x)) else np.nan for x in x_vals])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, y_vals, label='f(x)', color='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Animación del Método de Intervalos por la Mitad')
    ax.grid(True, linestyle=':', alpha=0.7)

    y_lim = ax.get_ylim()

    # Inicializar elementos de la gráfica
    point_x1, = ax.plot([], [], 'ro', markersize=8, label='x1')
    point_xm, = ax.plot([], [], 'bo', markersize=8, label='x_m')
    point_x2, = ax.plot([], [], 'go', markersize=8, label='x2')
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
    iter_text = ax.text((a+b)/2, y_lim[0]*1.05, '', color='darkgreen', ha='center', va='bottom', 
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    ax.legend()

    def init():
        point_x1.set_data([], [])
        point_xm.set_data([], [])
        point_x2.set_data([], [])
        eliminated_span.set_width(0)
        new_span.set_width(0)
        iter_text.set_text('')
        return point_x1, point_xm, point_x2, eliminated_span, new_span, iter_text

    def update(frame):
        if frame == 0:
            # Mostrar solo el intervalo inicial
            point_x1.set_data([], [])
            point_xm.set_data([], [])
            point_x2.set_data([], [])
            eliminated_span.set_width(0)
            new_span.set_width(0)
            iter_text.set_text('Iteración 0\nIntervalo Inicial')
        else:
            iter_data = iteraciones[frame - 1]
            a_new, b_new = iter_data['a'], iter_data['b']
            x1, xm, x2 = iter_data['x1'], iter_data['x_m'], iter_data['x2']
            f_x1, f_xm, f_x2 = iter_data['f(x1)'], iter_data['f(x_m)'], iter_data['f(x2)']
            
            # Actualizar puntos
            point_x1.set_data([x1], [f_x1])
            point_xm.set_data([xm], [f_xm])
            point_x2.set_data([x2], [f_x2])
            
            # Actualizar líneas de límites
            line_a.set_xdata([a_new])
            text_a.set_position((a_new, y_lim[1]*0.95))
            text_a.set_text(f'a={a_new:.2f}')
            line_b.set_xdata([b_new])
            text_b.set_position((b_new, y_lim[1]*0.95))
            text_b.set_text(f'b={b_new:.2f}')
            
            # Actualizar regiones
            if frame == 1:
                eliminated_span.set_width(0)
            elif a_new > a:
                eliminated_span.set_xy((a, y_lim[0]))
                eliminated_span.set_width(a_new - a)
            elif b_new < b:
                eliminated_span.set_xy((b_new, y_lim[0]))
                eliminated_span.set_width(b - b_new)
            else:
                eliminated_span.set_width(0)
            
            new_span.set_xy((a_new, y_lim[0]))
            new_span.set_width(b_new - a_new)
            
            iter_text.set_text(f'Iteración {frame}\nIntervalo [{a_new:.2f}, {b_new:.2f}]')
        
        return point_x1, point_xm, point_x2, line_a, text_a, line_b, text_b, eliminated_span, new_span, iter_text

    # Crear la animación (un fotograma por iteración más el inicial)
    ani = animation.FuncAnimation(fig, update, frames=len(iteraciones) + 1, init_func=init, interval=1000)

    # Guardar la animación como GIF
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

def show_intervalos_mitad(FUNCIONES, evaluar_funcion):
    """
    Interfaz de Streamlit para el método de intervalos por la mitad
    """
    st.markdown("""
    ## Método de Intervalos por la Mitad
    
    Este método divide el intervalo de búsqueda por la mitad en cada iteración, 
    evaluando puntos estratégicos para determinar dónde se encuentra el mínimo.
    """)
    
    # Crear dos columnas para los parámetros
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Selección de Función")
        funcion_seleccionada = st.selectbox(
            "Elige una función:",
            list(FUNCIONES.keys()),
            key="intervalos_funcion"
        )
        
        # Mostrar la función seleccionada
        st.latex(FUNCIONES[funcion_seleccionada]["latex"])
        st.write(f"**Dominio:** {FUNCIONES[funcion_seleccionada]['dominio']}")
    
    with col2:
        st.subheader("Parámetros del Método")
        
        # Obtener intervalo sugerido para la función seleccionada
        intervalo_sugerido = FUNCIONES[funcion_seleccionada]["intervalos"][0]
        a_sugerido, b_sugerido = intervalo_sugerido[0], intervalo_sugerido[1]
        
        a = st.number_input(
            "Límite inferior (a):",
            value=float(a_sugerido),
            format="%.4f",
            key="intervalos_a"
        )
        
        b = st.number_input(
            "Límite superior (b):",
            value=float(b_sugerido),
            format="%.4f",
            key="intervalos_b"
        )
        
        epsilon = st.number_input(
            "Tolerancia (ε):",
            value=0.01,
            min_value=0.0001,
            max_value=1.0,
            format="%.4f",
            key="intervalos_epsilon"
        )
    
    # Validar que a < b
    if a >= b:
        st.error("El límite inferior debe ser menor que el límite superior.")
        return
    
    # Botón para ejecutar el método
    if st.button("Ejecutar Método de Intervalos por la Mitad", key="ejecutar_intervalos"):
        try:
            # Crear función lambda para evaluar
            func = lambda x: evaluar_funcion(x, funcion_seleccionada)
            
            # Ejecutar el método
            with st.spinner("Ejecutando método de intervalos por la mitad..."):
                x_optimo, f_optimo, iteraciones = interval_halving(func, a, b, epsilon)
            
            # Mostrar animación
            st.subheader("Animación del Proceso")
            gif_output = plot_interval_halving_animation(func, a, b, iteraciones, epsilon)
            st.image(gif_output, caption="Progreso del algoritmo por iteración", use_container_width=True)
            
            # Mostrar resultados
            st.success("¡Método ejecutado exitosamente!")
            
            # Crear tres columnas para mostrar resultados
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Punto Óptimo (x*)", f"{x_optimo:.6f}")
            
            with col2:
                st.metric("Valor Mínimo f(x*)", f"{f_optimo:.6f}")
            
            with col3:
                st.metric("Número de Iteraciones", len(iteraciones))
            
            # Mostrar tabla de iteraciones
            st.subheader("Tabla de Iteraciones")
            df_iteraciones = pd.DataFrame(iteraciones)
            
            # Formatear números para mejor visualización
            numeric_cols = ['a', 'b', 'L', 'x_m', 'x1', 'x2', 'f(x_m)', 'f(x1)', 'f(x2)']
            for col in numeric_cols:
                df_iteraciones[col] = df_iteraciones[col].round(6)
            
            st.dataframe(df_iteraciones, use_container_width=True)
            
            # Crear gráfico de la función y el punto óptimo
            st.subheader("Visualización de la Función y Punto Óptimo")
            
            # Generar puntos para la gráfica
            x_vals = np.linspace(a, b, 1000)
            y_vals = []
            
            for x in x_vals:
                try:
                    y_vals.append(func(x))
                except:
                    y_vals.append(np.nan)
            
            # Crear gráfico con Matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Graficar la función
            ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
            
            # Marcar el punto óptimo
            ax.plot(x_optimo, f_optimo, 'r*', markersize=15, 
                   label=f'Mínimo (x*={x_optimo:.4f}, f(x*)={f_optimo:.4f})')
            
            # Agregar línea vertical en el punto óptimo
            ax.axvline(x=x_optimo, color='red', linestyle='--', alpha=0.7,
                      label=f'x* = {x_optimo:.4f}')
            
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title(f'Función {funcion_seleccionada} - Método de Intervalos por la Mitad')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Mostrar el gráfico en Streamlit
            st.pyplot(fig)
            plt.close()
            
            # Mostrar convergencia
            st.subheader("Convergencia del Método")
            
            # Gráfico de convergencia del intervalo
            longitudes = [iter_data['L'] for iter_data in iteraciones]
            iteraciones_num = list(range(1, len(longitudes) + 1))
            
            fig_conv, ax_conv = plt.subplots(figsize=(10, 6))
            
            ax_conv.plot(iteraciones_num, longitudes, 'g-o', linewidth=2, 
                        markersize=6, label='Longitud del Intervalo')
            ax_conv.axhline(y=epsilon, color='red', linestyle='--', 
                           label=f'Tolerancia ε = {epsilon}')
            
            ax_conv.set_xlabel('Iteración')
            ax_conv.set_ylabel('Longitud del Intervalo (L)')
            ax_conv.set_title('Convergencia - Reducción de la Longitud del Intervalo')
            ax_conv.legend()
            ax_conv.grid(True, alpha=0.3)
            
            # Mostrar el gráfico de convergencia
            st.pyplot(fig_conv)
            plt.close()
            
            # Mostrar análisis de convergencia
            st.subheader("Análisis de Convergencia")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Reducción del Intervalo por Iteración:**")
                reduccion_datos = []
                for i, iter_data in enumerate(iteraciones):
                    if i == 0:
                        reduccion = 0
                    else:
                        reduccion = ((iteraciones[i-1]['L'] - iter_data['L']) / iteraciones[i-1]['L']) * 100
                    reduccion_datos.append({
                        'Iteración': i + 1,
                        'Longitud L': iter_data['L'],
                        'Reducción (%)': reduccion
                    })
                
                df_reduccion = pd.DataFrame(reduccion_datos)
                st.dataframe(df_reduccion.round(4), use_container_width=True)
            
            with col2:
                st.write("**Estadísticas de Convergencia:**")
                longitud_inicial = iteraciones[0]['L']
                longitud_final = iteraciones[-1]['L']
                reduccion_total = ((longitud_inicial - longitud_final) / longitud_inicial) * 100
                
                st.metric("Longitud Inicial", f"{longitud_inicial:.6f}")
                st.metric("Longitud Final", f"{longitud_final:.6f}")
                st.metric("Reducción Total", f"{reduccion_total:.2f}%")
                st.metric("Tasa de Convergencia", f"{reduccion_total/len(iteraciones):.2f}%/iter")
            
        except Exception as e:
            st.error(f"Error durante la ejecución: {str(e)}")
    
    # Información adicional sobre el método
    with st.expander("ℹ️ Información sobre el Método de Intervalos por la Mitad"):
        st.markdown("""
        ### Descripción del Algoritmo
        
        El método de intervalos por la mitad es una técnica de optimización unidimensional que:
        
        1. **Divide el intervalo**: En cada iteración, evalúa tres puntos estratégicos
        2. **Compara valores**: Determina cuál subintervalo contiene el mínimo
        3. **Reduce el espacio**: Elimina la mitad del intervalo que no contiene el óptimo
        4. **Converge**: Repite hasta que el intervalo sea menor que la tolerancia
        
        ### Puntos de Evaluación
        - **x_m**: Punto medio del intervalo = (a + b) / 2
        - **x1**: Primer cuarto = a + L/4
        - **x2**: Tercer cuarto = b - L/4
        
        ### Reglas de Reducción
        - Si f(x1) < f(x_m): El mínimo está en [a, x_m] → b = x_m
        - Si f(x2) < f(x_m): El mínimo está en [x_m, b] → a = x_m  
        - Si no: El mínimo está en [x1, x2] → a = x1, b = x2
        
        ### Ventajas
        - ✅ Convergencia garantizada para funciones unimodales
        - ✅ Reducción sistemática del espacio de búsqueda
        - ✅ Fácil implementación y comprensión
        - ✅ No requiere derivadas
        
        ### Desventajas
        - ❌ Requiere que la función sea unimodal en el intervalo
        - ❌ Puede ser más lento que otros métodos como la sección dorada
        - ❌ Necesita evaluar la función tres veces por iteración
        
        ### Complejidad
        - **Temporal**: O(log(L₀/ε)) donde L₀ es la longitud inicial y ε la tolerancia
        - **Espacial**: O(1) - usa espacio constante
        """)