import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import pandas as pd
import tempfile
import io
import os

def fibonacci_search(f, a, b, n):
    """
    Método de búsqueda de Fibonacci para optimización unidimensional
    """
    # Generar secuencia de Fibonacci hasta Fn+1
    F = [0, 1]
    for i in range(2, n + 2):
        F.append(F[i-1] + F[i-2])
    
    L = b - a
    k = 2
    iteraciones = []
    
    # Paso 2 - Inicialización
    Lk = (F[n - k + 1] / F[n + 1]) * L
    x1 = a + Lk
    x2 = b - Lk
    f_x1 = f(x1)
    f_x2 = f(x2)
    
    # Guardar información inicial
    iteraciones.append({
        'Iteración': 1,
        'k': k,
        'a': a,
        'b': b,
        'L': b - a,
        'Lk': Lk,
        'x1': x1,
        'x2': x2,
        'f(x1)': f_x1,
        'f(x2)': f_x2,
        'F[n-k+1]': F[n - k + 1],
        'F[n+1]': F[n + 1],
        'Decisión': 'Inicialización'
    })
    
    iteration = 2
    
    while k < n:
        decision = ""
        if f_x1 < f_x2:
            # Eliminar el subintervalo [x2, b]
            decision = f"f(x1)={f_x1:.6f} < f(x2)={f_x2:.6f} → Eliminar [x2, b]"
            b = x2
            x2 = x1
            f_x2 = f_x1
            k += 1
            Lk = (F[n - k + 1] / F[n + 1]) * (b - a)
            x1 = a + Lk
            f_x1 = f(x1)
        else:
            # Eliminar el subintervalo [a, x1]
            decision = f"f(x1)={f_x1:.6f} ≥ f(x2)={f_x2:.6f} → Eliminar [a, x1]"
            a = x1
            x1 = x2
            f_x1 = f_x2
            k += 1
            Lk = (F[n - k + 1] / F[n + 1]) * (b - a)
            x2 = b - Lk
            f_x2 = f(x2)
        
        # Guardar información de la iteración
        iteraciones.append({
            'Iteración': iteration,
            'k': k,
            'a': a,
            'b': b,
            'L': b - a,
            'Lk': Lk,
            'x1': x1,
            'x2': x2,
            'f(x1)': f_x1,
            'f(x2)': f_x2,
            'F[n-k+1]': F[n - k + 1],
            'F[n+1]': F[n + 1],
            'Decisión': decision
        })
        
        iteration += 1
    
    # Terminar: devolver el punto medio del intervalo final
    x_optimo = (a + b) / 2
    return x_optimo, f(x_optimo), iteraciones, F

def plot_fibonacci_search_animation(f, a, b, iteraciones):
    """
    Crea una animación mostrando el proceso del método de búsqueda de Fibonacci
    """
    x_plot_min = min(a, min(iter_data['a'] for iter_data in iteraciones)) - (b - a) * 0.1
    x_plot_max = max(b, max(iter_data['b'] for iter_data in iteraciones)) + (b - a) * 0.1
    x_vals = np.linspace(x_plot_min, x_plot_max, 500)
    y_vals = np.array([f(x) if not np.isnan(f(x)) else np.nan for x in x_vals])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, y_vals, label='f(x)', color='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Animación del Método de Búsqueda de Fibonacci')
    ax.grid(True, linestyle=':', alpha=0.7)

    y_lim = ax.get_ylim()

    # Inicializar elementos de la gráfica
    point_x1, = ax.plot([], [], 'go', markersize=8, label='x1')
    point_x2, = ax.plot([], [], 'mo', markersize=8, label='x2')
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
        point_x2.set_data([], [])
        eliminated_span.set_width(0)
        new_span.set_width(0)
        iter_text.set_text('')
        return point_x1, point_x2, eliminated_span, new_span, iter_text

    def update(frame):
        if frame == 0:
            # Mostrar solo el intervalo inicial
            point_x1.set_data([], [])
            point_x2.set_data([], [])
            eliminated_span.set_width(0)
            new_span.set_width(0)
            iter_text.set_text('Iteración 0\nIntervalo Inicial')
        else:
            iter_data = iteraciones[frame - 1]
            a_new, b_new = iter_data['a'], iter_data['b']
            x1, x2 = iter_data['x1'], iter_data['x2']
            f_x1, f_x2 = iter_data['f(x1)'], iter_data['f(x2)']
            
            # Actualizar puntos
            point_x1.set_data([x1], [f_x1])
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
        
        return point_x1, point_x2, line_a, text_a, line_b, text_b, eliminated_span, new_span, iter_text

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

def show_fibonacci_search(FUNCIONES, evaluar_funcion):
    """
    Interfaz de Streamlit para el método de búsqueda de Fibonacci
    """
    st.markdown("""
    ## Método de Búsqueda de Fibonacci
    
    Este método utiliza la secuencia de Fibonacci para determinar los puntos de evaluación,
    optimizando el número de evaluaciones de función necesarias.
    """)
    
    # Crear dos columnas para los parámetros
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Selección de Función")
        funcion_seleccionada = st.selectbox(
            "Elige una función:",
            list(FUNCIONES.keys()),
            key="fibonacci_funcion"
        )
        
        st.latex(FUNCIONES[funcion_seleccionada]["latex"])
        st.write(f"**Dominio:** {FUNCIONES[funcion_seleccionada]['dominio']}")
    
    with col2:
        st.subheader("Parámetros del Método")
        
        intervalo_sugerido = FUNCIONES[funcion_seleccionada]["intervalos"][0]
        a_sugerido, b_sugerido = intervalo_sugerido[0], intervalo_sugerido[1]
        
        a = st.number_input(
            "Límite inferior (a):",
            value=float(a_sugerido),
            format="%.4f",
            key="fibonacci_a"
        )
        
        b = st.number_input(
            "Límite superior (b):",
            value=float(b_sugerido),
            format="%.4f",
            key="fibonacci_b"
        )
        
        n = st.number_input(
            "Número de términos de Fibonacci (n):",
            value=10,
            min_value=3,
            max_value=20,
            step=1,
            key="fibonacci_n",
            help="Determina la precisión del método. Más términos = mayor precisión"
        )
    
    # Validar que a < b
    if a >= b:
        st.error("El límite inferior debe ser menor que el límite superior.")
        return
    
    # Mostrar información sobre la secuencia de Fibonacci
    with st.expander("🔢 Visualizar Secuencia de Fibonacci"):
        F_preview = [0, 1]
        for i in range(2, n + 2):
            F_preview.append(F_preview[i-1] + F_preview[i-2])
        
        st.write(f"**Secuencia de Fibonacci hasta F{n+1}:**")
        
        # Crear DataFrame para mostrar la secuencia
        df_fib = pd.DataFrame({
            'Índice': list(range(len(F_preview))),
            'F(n)': F_preview,
            'Razón F(n)/F(n-1)': ['-'] + ['-'] + [F_preview[i]/F_preview[i-1] if F_preview[i-1] != 0 else '-' for i in range(2, len(F_preview))]
        })
        
        st.dataframe(df_fib, use_container_width=True)
        
        # Calcular la razón áurea aproximada
        if len(F_preview) > 3:
            ratio_aurea = F_preview[-1] / F_preview[-2]
            st.write(f"**Aproximación a la Razón Áurea (φ):** {ratio_aurea:.6f}")
            st.write(f"**Razón Áurea exacta:** {(1 + np.sqrt(5))/2:.6f}")
    
    # Botón para ejecutar el método
    if st.button("Ejecutar Método de Búsqueda de Fibonacci", key="ejecutar_fibonacci"):
        try:
            # Crear función lambda para evaluar
            func = lambda x: evaluar_funcion(x, funcion_seleccionada)
            
            # Ejecutar el método
            with st.spinner("Ejecutando método de búsqueda de Fibonacci..."):
                x_optimo, f_optimo, iteraciones, F = fibonacci_search(func, a, b, n)
            
            # Mostrar animación
            st.subheader("Animación del Proceso")
            gif_output = plot_fibonacci_search_animation(func, a, b, iteraciones)
            st.image(gif_output, caption="Progreso del algoritmo por iteración", use_container_width=True)
            
            # Mostrar resultados
            st.success("¡Método ejecutado exitosamente!")
            
            # Crear columnas para mostrar resultados principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Punto Óptimo (x*)", f"{x_optimo:.6f}")
            
            with col2:
                st.metric("Valor Mínimo f(x*)", f"{f_optimo:.6f}")
            
            with col3:
                st.metric("Número de Iteraciones", len(iteraciones))
            
            with col4:
                st.metric("Evaluaciones de f(x)", len(iteraciones) + 1)
            
            # Mostrar información de la secuencia utilizada
            st.subheader("Información de la Secuencia de Fibonacci")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Términos de Fibonacci utilizados:**")
                fib_info = pd.DataFrame({
                    'Índice': list(range(len(F))),
                    'F(n)': F
                })
                st.dataframe(fib_info, use_container_width=True)
            
            with col2:
                st.write("**Estadísticas:**")
                st.write(f"F({n+1}) = {F[n+1]}")
                st.write(f"Longitud inicial: {b-a:.6f}")
                st.write(f"Longitud final: {iteraciones[-1]['L']:.6f}")
                reduccion = ((b-a - iteraciones[-1]['L']) / (b-a)) * 100
                st.write(f"Reducción total: {reduccion:.2f}%")
            
            # Mostrar tabla de iteraciones
            st.subheader("Tabla Detallada de Iteraciones")
            df_iteraciones = pd.DataFrame(iteraciones)
            
            # Formatear números para mejor visualización
            numeric_cols = ['a', 'b', 'L', 'Lk', 'x1', 'x2', 'f(x1)', 'f(x2)']
            for col in numeric_cols:
                if col in df_iteraciones.columns:
                    df_iteraciones[col] = df_iteraciones[col].round(6)
            
            # Reorganizar columnas para mejor legibilidad
            cols_order = ['Iteración', 'k', 'a', 'b', 'L', 'x1', 'x2', 'f(x1)', 'f(x2)', 'F[n-k+1]', 'F[n+1]', 'Decisión']
            df_display = df_iteraciones[cols_order]
            
            st.dataframe(df_display, use_container_width=True)
            
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
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Graficar la función
            ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
            
            # Marcar el punto óptimo
            ax.plot(x_optimo, f_optimo, 'r*', markersize=15, 
                   label=f'Mínimo (x*={x_optimo:.4f}, f(x*)={f_optimo:.4f})')
            
            # Marcar los puntos de evaluación de la última iteración
            if iteraciones:
                ultima_iter = iteraciones[-1]
                ax.plot(ultima_iter['x1'], ultima_iter['f(x1)'], 'go', markersize=8, 
                       label=f"x1 = {ultima_iter['x1']:.4f}")
                ax.plot(ultima_iter['x2'], ultima_iter['f(x2)'], 'mo', markersize=8, 
                       label=f"x2 = {ultima_iter['x2']:.4f}")
            
            # Agregar línea vertical en el punto óptimo
            ax.axvline(x=x_optimo, color='red', linestyle='--', alpha=0.7)
            
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title(f'Función {funcion_seleccionada} - Método de Búsqueda de Fibonacci')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Mostrar el gráfico en Streamlit
            st.pyplot(fig)
            plt.close()
            
            # Mostrar convergencia
            st.subheader("Análisis de Convergencia")
            
            # Gráfico de convergencia del intervalo
            longitudes = [iter_data['L'] for iter_data in iteraciones]
            iteraciones_num = list(range(1, len(longitudes) + 1))
            
            fig_conv, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Gráfico 1: Reducción de longitud
            ax1.plot(iteraciones_num, longitudes, 'g-o', linewidth=2, 
                    markersize=6, label='Longitud del Intervalo')
            ax1.set_xlabel('Iteración')
            ax1.set_ylabel('Longitud del Intervalo (L)')
            ax1.set_title('Convergencia - Reducción de la Longitud del Intervalo')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: Evolución de los puntos x1 y x2
            x1_vals = [iter_data['x1'] for iter_data in iteraciones]
            x2_vals = [iter_data['x2'] for iter_data in iteraciones]
            
            ax2.plot(iteraciones_num, x1_vals, 'b-o', linewidth=2, 
                    markersize=6, label='x1')
            ax2.plot(iteraciones_num, x2_vals, 'r-o', linewidth=2, 
                    markersize=6, label='x2')
            ax2.axhline(y=x_optimo, color='black', linestyle='--', alpha=0.7,
                       label=f'x* = {x_optimo:.4f}')
            ax2.set_xlabel('Iteración')
            ax2.set_ylabel('Valor de x')
            ax2.set_title('Evolución de los Puntos de Evaluación')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Mostrar el gráfico de convergencia
            st.pyplot(fig_conv)
            plt.close()
            
            # Análisis de eficiencia
            st.subheader("Análisis de Eficiencia")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Comparación con otros métodos:**")
                evaluaciones_fibonacci = len(iteraciones) + 1
                evaluaciones_teoricas_biseccion = int(np.log2((b-a)/iteraciones[-1]['L'])) + 1
                evaluaciones_teoricas_dorada = int(np.log((b-a)/iteraciones[-1]['L'])/np.log((1+np.sqrt(5))/2)) + 2
                
                comparison_data = {
                    'Método': ['Fibonacci', 'Bisección (teórica)', 'Sección Dorada (teórica)'],
                    'Evaluaciones': [evaluaciones_fibonacci, evaluaciones_teoricas_biseccion, evaluaciones_teoricas_dorada],
                    'Eficiencia': ['Óptima', 'Menos eficiente', 'Casi óptima']
                }
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
            
            with col2:
                st.write("**Propiedades del método:**")
                st.write(f"• **Número de Fibonacci usado:** F({n+1}) = {F[n+1]}")
                st.write(f"• **Reducción de incertidumbre:** 1/{F[n+1]}")
                st.write(f"• **Precisión final:** {iteraciones[-1]['L']:.6f}")
                st.write(f"• **Tasa de convergencia:** {(longitudes[0]/longitudes[-1]):.2f}:1")
                
        except Exception as e:
            st.error(f"Error durante la ejecución: {str(e)}")
    
    # Información adicional sobre el método
    with st.expander("ℹ️ Información sobre el Método de Búsqueda de Fibonacci"):
        st.markdown("""
        ### Descripción del Algoritmo
        
        El método de búsqueda de Fibonacci es una técnica de optimización que utiliza la famosa secuencia de Fibonacci para determinar los puntos de evaluación de manera óptima.
        
        ### Secuencia de Fibonacci
        - F₀ = 0, F₁ = 1
        - Fₙ = Fₙ₋₁ + Fₙ₋₂ para n ≥ 2
        - **Propiedad clave:** Fₙ₊₁ = Fₙ + Fₙ₋₁
        
        ### Algoritmo paso a paso:
        
        1. **Inicialización:**
           - Generar secuencia de Fibonacci hasta F_{n+1}
           - Calcular L₀ = b - a (longitud inicial)
           - k = 2 (contador de iteraciones)
        
        2. **Cálculo de puntos:**
           - Lₖ = (F_{n-k+1} / F_{n+1}) × L
           - x₁ = a + Lₖ
           - x₂ = b - Lₖ
        
        3. **Comparación y reducción:**
           - Si f(x₁) < f(x₂): eliminar [x₂, b], actualizar b = x₂
           - Si f(x₁) ≥ f(x₂): eliminar [a, x₁], actualizar a = x₁
           - Incrementar k y recalcular puntos
        
        4. **Terminación:** Cuando k = n, devolver (a + b) / 2
        
        ### Ventajas del Método de Fibonacci
        
        ✅ **Número mínimo de evaluaciones:** Es matemáticamente óptimo  
        ✅ **Convergencia predecible:** El número de iteraciones es exactamente n-2  
        ✅ **Alta eficiencia:** Reduce la incertidumbre en factor 1/F_{n+1}  
        ✅ **No requiere derivadas:** Solo evaluaciones de la función  
        ✅ **Convergencia garantizada:** Para funciones unimodales  
        
        ### Desventajas
        
        ❌ **Número fijo de iteraciones:** Debe especificarse n de antemano  
        ❌ **Requiere unimodalidad:** La función debe tener un solo mínimo  
        ❌ **Menos flexible:** No se puede parar antes si se alcanza precisión deseada  
        
        ### Relación con la Razón Áurea
        
        Cuando n → ∞, la razón F_{n+1}/F_n → φ = (1 + √5)/2 ≈ 1.618 (razón áurea)
        
        Esto conecta el método de Fibonacci con el método de la Sección Dorada, donde φ se usa directamente.
        
        ### Complejidad
        - **Temporal:** O(n) donde n es el número de términos de Fibonacci
        - **Espacial:** O(n) para almacenar la secuencia de Fibonacci
        - **Evaluaciones de función:** Exactamente n+1 evaluaciones
        """)