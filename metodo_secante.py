import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import io

def df(func, x, h=1e-6):
    """Calcula la derivada numérica de una función en un punto x"""
    if abs(x) < h:
        return (func(2*h) - func(0)) / (2*h)
    return (func(x + h) - func(x - h)) / (2 * h)

def bounding_phase(f, x0, delta=0.1, max_iter=1000):
    """Algoritmo Bounding-Phase para encontrar el intervalo que contiene el óptimo"""
    x = max(x0, 0.01)  
    df_x = df(f, x)
    
    if df_x > 0: 
        delta = -delta
    
    x_prev = x
    x = x + delta
    
    if x <= 0 and delta < 0:
        x = 0.001  
    
    iter_count = 0
    iteraciones = [(x_prev, f(x_prev), df_x)]
    
    while df(f, x_prev) * df(f, x) > 0 and iter_count < max_iter:
        delta = 2 * delta
        x_prev = x
        x = x + delta
        
        if x <= 0 and delta < 0:
            x = 0.001 
            break
        
        iteraciones.append((x, f(x), df(f, x)))
        iter_count += 1
    
    if x < x_prev:
        return [x, x_prev], iteraciones
    else:
        return [x_prev, x], iteraciones

def secant_method(f, interval, epsilon=0.001, max_iter=100):
    """Método de la Secante para optimización"""
    x1, x2 = interval
    points = [x1, x2] 
    iteraciones = []
    
    iter_count = 0
    
    while abs(x2 - x1) > epsilon and iter_count < max_iter:
        df1 = df(f, x1)
        df2 = df(f, x2)
        
        # Guardar información de la iteración
        iteraciones.append({
            'iteracion': iter_count + 1,
            'x1': x1,
            'x2': x2,
            'f_x1': f(x1),
            'f_x2': f(x2),
            'df_x1': df1,
            'df_x2': df2,
            'diferencia': abs(x2 - x1)
        })
        
        # Aumentar la tolerancia para evitar salida prematura
        if abs(df2 - df1) < 1e-8:  # Cambiado de 1e-10 a 1e-8
            st.write(f"**Depuración** - Salida en iteración {iter_count + 1}: Pendiente |df2 - df1| = {abs(df2 - df1)} < 1e-8")
            break
            
        try:
            z = x2 - df2 / ((df2 - df1) / (x2 - x1))
            
            if np.isnan(z) or np.isinf(z):
                st.write(f"**Depuración** - Salida en iteración {iter_count + 1}: z es NaN o infinito (z = {z})")
                break
                
            if z <= 0 and x1 > 0 and x2 > 0:
                z = min(x1, x2) / 2 
                st.write(f"**Depuración** - Ajuste en iteración {iter_count + 1}: z = {z} (evitando z <= 0)")
                
            points.append(z)
            iteraciones[-1]['z'] = z  # Añadir z a la iteración
            iteraciones[-1]['f_z'] = f(z)
            iteraciones[-1]['df_z'] = df(f, z)
            
            # Verificar convergencia
            if abs(df(f, z)) < epsilon and len(points) >= 3:  # Asegurar al menos 3 puntos
                st.write(f"**Depuración** - Convergencia alcanzada en iteración {iter_count + 1}: |df(z)| = {abs(df(f, z))} < {epsilon}")
                return z, points, iteraciones
            
            x1 = x2
            x2 = z
            
        except Exception as e:
            st.write(f"**Depuración** - Error en iteración {iter_count + 1}: {str(e)}")
            break
            
        iter_count += 1
    
    st.write(f"**Depuración** - Fin del método de la secante: x2 = {x2}, |x2 - x1| = {abs(x2 - x1)}, iteraciones = {iter_count}")
    return x2, points, iteraciones

def plot_function_and_secant(f, points, domain, title="Método de la Secante"):
    """Visualiza la función y los puntos visitados por el método de la secante"""
    x = np.linspace(domain[0], domain[1], 1000)
    y = [f(xi) for xi in x]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plotear la función
    ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
    
    # Plotear los puntos visitados
    y_points = [f(p) for p in points]
    ax.plot(points, y_points, 'ro-', markersize=8, linewidth=2, label='Puntos visitados')
    
    # Destacar la solución final
    ax.plot(points[-1], y_points[-1], 'g*', markersize=15, label=f'Solución: x = {points[-1]:.6f}')
    
    # Añadir líneas secantes
    for i in range(len(points)-1):
        if i < len(points)-2:
            x_vals = [points[i], points[i+1]]
            y_vals = [f(points[i]), f(points[i+1])]
            ax.plot(x_vals, y_vals, 'r--', alpha=0.5, linewidth=1)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    return fig

def plot_secant_animation(f, points, domain, title, fps=1):
    """
    Crea una animación mostrando el proceso del método de la secante.
    """
    try:
        # Depuración
        st.write(f"**Depuración** - Tamaño del historial: {len(points)}")
        st.write(f"**Depuración** - Puntos del historial: {points}")
        st.write(f"**Depuración** - Dominio: {domain}")

        if len(points) < 3:
            st.warning("El historial tiene menos de 3 puntos. La animación requiere al menos 3 puntos (x1, x2, z).")
            return None

        # Crear malla para la gráfica
        x_vals = np.linspace(domain[0], domain[1], 100)
        y_vals = []
        for x in x_vals:
            try:
                y_vals.append(f(x))
            except:
                y_vals.append(np.nan)

        if np.all(np.isnan(y_vals)):
            st.error("Error: Todos los valores de la función son NaN o infinitos.")
            return None

        # Crear la figura
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Animación del Método de la Secante - {title}')
        ax.grid(True, alpha=0.3)

        # Inicializar elementos
        points_line, = ax.plot([], [], 'ro', markersize=5, label='Puntos visitados')
        x1_point, = ax.plot([], [], 'ro', markersize=8, label='x1')
        x2_point, = ax.plot([], [], 'ro', markersize=8, label='x2')
        final_point, = ax.plot([], [], '*', color='yellow', markersize=12, label='Punto óptimo')
        secant_line, = ax.plot([], [], 'r--', linewidth=1.5, label='Línea secante')
        iter_text = ax.text(
            x_vals[0] + 0.05*(x_vals[-1] - x_vals[0]),
            min(y_vals) + 0.05*(max(y_vals) - min(y_vals)),
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
            x1_point.set_data([], [])
            x2_point.set_data([], [])
            final_point.set_data([], [])
            secant_line.set_data([], [])
            iter_text.set_text('')
            return points_line, x1_point, x2_point, final_point, secant_line, iter_text

        def update(frame):
            x_points = points[:frame+1]
            y_points = [f(p) for p in x_points]
            points_line.set_data(x_points, y_points)

            if frame < len(points) - 1:
                x1 = points[frame]
                x2 = points[frame + 1 if frame + 1 < len(points) else frame]
                x1_point.set_data([x1], [f(x1)])
                x2_point.set_data([x2], [f(x2)])
                final_point.set_data([], [])
                # Línea secante
                secant_line.set_data([x1, x2], [f(x1), f(x2)])
                iter_text.set_text(f'Iteración {frame}\nx1 = {x1:.4f}\nx2 = {x2:.4f}\nError = {abs(x2 - x1):.4e}')
            else:
                x1_point.set_data([], [])
                x2_point.set_data([], [])
                final_point.set_data([points[-1]], [f(points[-1])])
                secant_line.set_data([], [])
                iter_text.set_text(f'Óptimo encontrado\nx* = {points[-1]:.4f}')

            return points_line, x1_point, x2_point, final_point, secant_line, iter_text

        # Crear la animación
        st.write("**Depuración** - Creando animación...")
        ani = animation.FuncAnimation(fig, update, frames=len(points), init_func=init, interval=1000//fps, blit=True)

        # Guardar la animación en un archivo temporal
        os.makedirs("./animations", exist_ok=True)
        temp_file_path = f"./animations/secant_animation_{np.random.randint(10000)}.gif"
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

        # Leer el archivo en BytesIO para mostrarlo
        with open(temp_file_path, 'rb') as file:
            output = io.BytesIO(file.read())
        
        plt.close(fig)
        output.seek(0)
        return output

    except Exception as e:
        st.error(f"Error al generar la animación: {str(e)}")
        return None

def show_metodo_secante(FUNCIONES, evaluar_funcion):
    """Función principal para mostrar el método de la secante en Streamlit"""
    
    st.markdown("""
    ## 📏 Método de la Secante para Optimización
    
    El método de la secante es una técnica iterativa para encontrar extremos de funciones que aproxima 
    la derivada usando líneas secantes entre puntos consecutivos. Es especialmente útil cuando no se 
    puede calcular la derivada analíticamente de forma sencilla.
    
    ### Características:
    - **Convergencia**: Superlineal (orden ≈ 1.618)
    - **Ventaja**: No requiere derivadas exactas
    - **Desventaja**: Puede ser menos estable que Newton-Raphson
    """)
    
    # Inicializar estado de sesión
    if 'secant_results' not in st.session_state:
        st.session_state.secant_results = None
        st.session_state.funcion_seleccionada = None
        st.session_state.fps = 1

    # Formulario para entrada de parámetros
    with st.form(key="secant_form"):
        st.subheader("⚙️ Configuración")
        col1, col2 = st.columns(2)
        
        with col1:
            funcion_seleccionada = st.selectbox(
                "Elige una función:",
                list(FUNCIONES.keys()),
                key="secante_funcion"
            )
            st.latex(FUNCIONES[funcion_seleccionada]["latex"])
            st.write(f"**Dominio:** {FUNCIONES[funcion_seleccionada]['dominio']}")
        
        with col2:
            intervalo_func = FUNCIONES[funcion_seleccionada]["intervalos"][0]
            dominio_min, dominio_max = intervalo_func[0], intervalo_func[1]
            
            punto_inicial = st.number_input(
                "Punto inicial (x₀):",
                min_value=float(dominio_min + 0.001),
                max_value=float(dominio_max - 0.001),
                value=float((dominio_min + dominio_max) / 2),
                step=0.1,
                format="%.6f",
                key="secante_x0"
            )
            
            epsilon = st.number_input(
                "Tolerancia (ε):",
                min_value=1e-10,
                max_value=1e-1,
                value=1e-3,  # Aumentado para evitar convergencia prematura
                step=1e-4,
                format="%.2e",
                key="secante_epsilon"
            )
            
            max_iter = st.number_input(
                "Máximo de iteraciones:",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                key="secante_max_iter"
            )
            
            delta_inicial = st.number_input(
                "Delta inicial (Bounding Phase):",
                min_value=0.001,
                max_value=1.0,
                value=0.5,  # Aumentado para un intervalo más amplio
                step=0.01,
                format="%.3f",
                key="secante_delta"
            )
            
            fps = st.slider(
                "Velocidad de la animación (FPS):",
                min_value=1,
                max_value=5,
                value=1,
                key="secant_fps"
            )
        
        submit_button = st.form_submit_button("🚀 Ejecutar Método de la Secante")

    # Ejecutar el algoritmo
    if submit_button:
        func = lambda x: evaluar_funcion(x, funcion_seleccionada)
        
        try:
            # Fase de acotamiento
            with st.spinner("Ejecutando fase de acotamiento..."):
                intervalo, iteraciones_bounding = bounding_phase(
                    func, punto_inicial, delta_inicial, max_iter
                )
            
            # Método de la secante
            with st.spinner("Ejecutando método de la secante..."):
                resultado, puntos, iteraciones = secant_method(
                    func, intervalo, epsilon, max_iter
                )
            
            st.session_state.secant_results = (resultado, puntos, iteraciones, intervalo, iteraciones_bounding)
            st.session_state.funcion_seleccionada = funcion_seleccionada
            st.session_state.fps = fps
            
            # Mostrar animación inmediatamente si hay suficientes puntos
            st.subheader("📽️ Animación del Proceso")
            with st.container():
                dominio_grafica = (FUNCIONES[funcion_seleccionada]["intervalos"][0][0], 
                                 FUNCIONES[funcion_seleccionada]["intervalos"][0][1])
                if len(puntos) >= 3:
                    gif_output = plot_secant_animation(
                        func, puntos, dominio_grafica, funcion_seleccionada, fps
                    )
                    if gif_output is not None:
                        st.image(gif_output, caption="Progreso del método de la secante", use_container_width=True, output_format='GIF')
                        st.success("Animación generada y mostrada correctamente.")
                        st.download_button(
                            label="📥 Descargar Animación",
                            data=gif_output,
                            file_name="secant_animation.gif",
                            mime="image/gif"
                        )
                    else:
                        st.error("No se pudo generar la animación. Revisa los mensajes de depuración arriba.")
                else:
                    st.warning(f"No se puede generar la animación: solo hay {len(puntos)} puntos. Se requieren al menos 3 puntos.")
        
        except Exception as e:
            st.error(f"❌ Error durante la ejecución: {str(e)}")
            st.write("**Posibles causas:**")
            st.write("- El punto inicial está fuera del dominio de la función")
            st.write("- La función no es diferenciable en el punto")
            st.write("- Los parámetros no son adecuados para esta función")
            return

    # Mostrar resultados adicionales si existen
    if st.session_state.secant_results:
        resultado, puntos, iteraciones, intervalo, iteraciones_bounding = st.session_state.secant_results
        func = lambda x: evaluar_funcion(x, st.session_state.funcion_seleccionada)
        fps = st.session_state.fps
        
        # Botón para regenerar animación
        with st.form(key="regen_animation_form"):
            st.subheader("🔄 Regenerar Animación")
            new_fps = st.slider("Cambiar velocidad de animación (FPS)", min_value=1, max_value=5, value=fps, key="regen_fps")
            regen_button = st.form_submit_button("🔄 Regenerar Animación")
            if regen_button and len(puntos) >= 3:
                st.session_state.fps = new_fps
                dominio_grafica = (FUNCIONES[st.session_state.funcion_seleccionada]["intervalos"][0][0], 
                                 FUNCIONES[st.session_state.funcion_seleccionada]["intervalos"][0][1])
                gif_output = plot_secant_animation(
                    func, puntos, dominio_grafica, st.session_state.funcion_seleccionada, new_fps
                )
                if gif_output is not None:
                    st.image(gif_output, caption="Progreso del método de la secante", use_container_width=True, output_format='GIF')
                    st.success("Animación regenerada correctamente.")
                    st.download_button(
                        label="📥 Descargar Animación",
                        data=gif_output,
                        file_name="secant_animation.gif",
                        mime="image/gif"
                    )
                else:
                    st.error("No se pudo regenerar la animación. Revisa los mensajes de depuración.")
            elif regen_button:
                st.warning(f"No se puede regenerar la animación: solo hay {len(puntos)} puntos. Se requieren al menos 3 puntos.")

        # Mostrar resultados
        st.markdown("---")
        st.subheader("📈 Resultados")
        
        st.success(f"✅ Intervalo encontrado: [{intervalo[0]:.6f}, {intervalo[1]:.6f}]")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Punto Óptimo", f"{resultado:.8f}")
        
        with col2:
            st.metric("📊 Valor de la Función", f"{func(resultado):.8f}")
        
        with col3:
            st.metric("🔄 Iteraciones", len(iteraciones))
        
        # Verificar si es mínimo o máximo
        derivada_final = df(func, resultado)
        st.metric("📐 Derivada Final", f"{derivada_final:.2e}")
        
        if abs(derivada_final) < epsilon:
            tipo_extremo = "🔴 Punto Crítico Encontrado"
            # Verificar segunda derivada para determinar tipo
            segunda_derivada = df(lambda x: df(func, x), resultado)
            if segunda_derivada > 0:
                tipo_extremo += " (Mínimo Local)"
            elif segunda_derivada < 0:
                tipo_extremo += " (Máximo Local)"
            else:
                tipo_extremo += " (Punto de Inflexión)"
                
            st.success(tipo_extremo)
        else:
            st.warning("⚠️ No se alcanzó la convergencia deseada")
        
        # Tabla de iteraciones
        if iteraciones:
            st.subheader("📋 Tabla de Iteraciones")
            df_iter = pd.DataFrame(iteraciones)
            df_iter = df_iter.round(8)
            st.dataframe(df_iter, use_container_width=True)
        
        # Gráfica estática
        st.subheader("📊 Visualización Estática")
        dominio_grafica = (FUNCIONES[st.session_state.funcion_seleccionada]["intervalos"][0][0], 
                          FUNCIONES[st.session_state.funcion_seleccionada]["intervalos"][0][1])
        fig = plot_function_and_secant(
            func, puntos, dominio_grafica, 
            f"Método de la Secante - {st.session_state.funcion_seleccionada}"
        )
        st.pyplot(fig)
        
        # Información adicional
        st.subheader("ℹ️ Información del Proceso")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Fase de Acotamiento:**")
            st.write(f"- Iteraciones: {len(iteraciones_bounding)}")
            st.write(f"- Intervalo inicial: [{dominio_grafica[0]}, {dominio_grafica[1]}]")
            st.write(f"- Intervalo final: [{intervalo[0]:.6f}, {intervalo[1]:.6f}]")
        
        with col2:
            st.write("**Método de la Secante:**")
            st.write(f"- Convergencia: {'✅ Sí' if abs(derivada_final) < epsilon else '❌ No'}")
            st.write(f"- Precisión alcanzada: {abs(derivada_final):.2e}")
            st.write(f"- Tolerancia requerida: {epsilon:.2e}")
        
        # Análisis de convergencia
        if len(puntos) > 2:
            st.subheader("📈 Análisis de Convergencia")
            
            errores = []
            for i in range(1, len(puntos)):
                error = abs(puntos[i] - puntos[i-1])
                errores.append(error)
            
            fig_conv, ax_conv = plt.subplots(figsize=(10, 6))
            ax_conv.semilogy(range(1, len(errores)+1), errores, 'bo-', linewidth=2, markersize=8)
            ax_conv.set_xlabel('Iteración', fontsize=12)
            ax_conv.set_ylabel('Error |x_{n+1} - x_n|', fontsize=12)
            ax_conv.set_title('Convergencia del Método de la Secante', fontsize=14, fontweight='bold')
            ax_conv.grid(True, alpha=0.3)
            
            st.pyplot(fig_conv)
    
    # Información teórica adicional
    with st.expander("📚 Información Teórica del Método de la Secante"):
        st.markdown("""
        ### Formulación Matemática
        
        El método de la secante para optimización busca encontrar puntos donde f'(x) = 0, utilizando la fórmula:
        
        $$x_{n+1} = x_n - \\frac{f'(x_n)}{\\frac{f'(x_n) - f'(x_{n-1})}{x_n - x_{n-1}}}$$
        
        ### Algoritmo:
        1. **Fase de Acotamiento**: Encuentra un intervalo que contenga el óptimo
        2. **Inicialización**: Toma dos puntos iniciales x₀ y x₁
        3. **Iteración**: Calcula el siguiente punto usando la fórmula de la secante
        4. **Convergencia**: Repite hasta que |x_{n+1} - x_n| < ε
        
        ### Ventajas:
        - No requiere el cálculo de segundas derivadas
        - Convergencia más rápida que el método de bisección
        - Relativamente simple de implementar
        
        ### Desventajas:
        - Puede no converger si la función no es bien comportada
        - Requiere dos puntos iniciales
        - Menos estable que Newton-Raphson
        
        ### Orden de Convergencia:
        El método de la secante tiene un orden de convergencia de aproximadamente 1.618 (número áureo φ).
        """)