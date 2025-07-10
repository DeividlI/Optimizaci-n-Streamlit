import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import pandas as pd
import tempfile
import io
import os

def golden_section_search(f, a, b, epsilon):
    """
    Implementa el método Golden Section Search
    """
    # Almacenar datos para mostrar el progreso
    iteraciones = []
    
    # Paso 1: Normalización
    a_w = 0
    b_w = 1
    L_w = 1
    k = 1
    w_to_x = lambda w: a + w * (b - a)
    
    # Paso 2: Inicializar w1, w2 y evaluar f(w1), f(w2)
    w1 = a_w + 0.618 * L_w
    w2 = b_w - 0.618 * L_w
    f_w1 = f(w_to_x(w1))
    f_w2 = f(w_to_x(w2))
    
    # Guardar primera iteración
    x1 = w_to_x(w1)
    x2 = w_to_x(w2)
    iteraciones.append({
        'Iteración': k,
        'a': a,
        'b': b,
        'x1': x1,
        'x2': x2,
        'f(x1)': f_w1,
        'f(x2)': f_w2,
        'L': b - a,
        'Eliminar': 'Inicio'
    })
    
    while True:
        # Usar la regla de eliminación de región
        if f_w1 < f_w2:
            # El mínimo está en [a_w, w2]
            b_w = w2
            # w2 se convierte en el nuevo w1, y su f(w2) también
            w2 = w1
            f_w2 = f_w1
            L_w = b_w - a_w
            w1 = a_w + 0.618 * L_w
            f_w1 = f(w_to_x(w1))
            eliminar = f"[{w_to_x(w2):.4f}, {b:.4f}]"
            b = w_to_x(w2)
        else:
            # El mínimo está en [w1, b_w]
            a_w = w1
            # w1 se convierte en el nuevo w2, y su f(w1) también
            w1 = w2
            f_w1 = f_w2
            L_w = b_w - a_w
            w2 = b_w - 0.618 * L_w
            f_w2 = f(w_to_x(w2))
            eliminar = f"[{a:.4f}, {w_to_x(w1):.4f}]"
            a = w_to_x(w1)
        
        k += 1
        
        # Guardar iteración actual
        x1 = w_to_x(w1)
        x2 = w_to_x(w2)
        iteraciones.append({
            'Iteración': k,
            'a': a,
            'b': b,
            'x1': x1,
            'x2': x2,
            'f(x1)': f_w1,
            'f(x2)': f_w2,
            'L': b - a,
            'Eliminar': eliminar
        })
        
        # Paso 3: Criterio de paro
        if abs(L_w * (b - a)) < epsilon:
            break
    
    # Convertimos el w medio a x real
    w_opt = (a_w + b_w) / 2
    x_opt = w_to_x(w_opt)
    
    return x_opt, f(x_opt), iteraciones

def plot_golden_section_animation(f, a, b, iteraciones, epsilon):
    """
    Crea una animación mostrando el proceso del método de Golden Section Search
    """
    x_plot_min = min(a, min(iter_data['a'] for iter_data in iteraciones)) - (b - a) * 0.1
    x_plot_max = max(b, max(iter_data['b'] for iter_data in iteraciones)) + (b - a) * 0.1
    x_vals = np.linspace(x_plot_min, x_plot_max, 500)
    y_vals = np.array([f(x) if not np.isnan(f(x)) else np.nan for x in x_vals])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, y_vals, label='f(x)', color='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Animación del Método de Golden Section Search')
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

def show_golden_section_search(FUNCIONES, evaluar_funcion):
    """
    Interfaz de Streamlit para el método Golden Section Search
    """
    st.markdown("""
    <div class="grid-background">
        <h2>🔍 Golden Section Search Method</h2>
        <p>Este método emplea la proporción áurea (φ ≈ 0.618) para una búsqueda eficiente del óptimo.</p>
        <p>La proporción áurea permite reducir el intervalo de búsqueda de manera óptima en cada iteración.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Selección de función
    col1, col2 = st.columns([1, 1])
    
    with col1:
        funcion_seleccionada = st.selectbox(
            "Selecciona la función a optimizar:",
            list(FUNCIONES.keys()),
            key="golden_funcion"
        )
        
        # Mostrar información de la función
        st.markdown("### Función seleccionada:")
        st.latex(FUNCIONES[funcion_seleccionada]["latex"])
        st.markdown(f"**Dominio:** {FUNCIONES[funcion_seleccionada]['dominio']}")
    
    with col2:
        st.markdown("### Parámetros del método:")
        
        # Obtener límites por defecto de la función
        intervalo_default = FUNCIONES[funcion_seleccionada]["intervalos"][0]
        a_default = intervalo_default[0]
        b_default = intervalo_default[1]
        
        a = st.number_input("Límite inferior (a):", value=float(a_default), key="golden_a")
        b = st.number_input("Límite superior (b):", value=float(b_default), key="golden_b")
        epsilon = st.number_input("Tolerancia (ε):", value=0.01, format="%.6f", key="golden_epsilon")
    
    if st.button("🔍 Ejecutar Golden Section Search", key="golden_ejecutar"):
        if a >= b:
            st.error("⚠️ El límite inferior debe ser menor que el superior")
            return
        
        try:
            with st.spinner("Ejecutando Golden Section Search..."):
                # Define la función lambda para el método
                def func(x):
                    return evaluar_funcion(x, funcion_seleccionada)
                
                # Ejecutar el método
                x_opt, f_opt, iteraciones = golden_section_search(func, a, b, epsilon)
            
            # Mostrar animación
            st.markdown("### Animación del Proceso")
            gif_output = plot_golden_section_animation(func, a, b, iteraciones, epsilon)
            st.image(gif_output, caption="Progreso del algoritmo por iteración", use_container_width=True)
            
            # Mostrar resultados
            st.success("✅ ¡Golden Section Search completado!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 x óptimo", f"{x_opt:.6f}")
            with col2:
                st.metric("📊 f(x) óptimo", f"{f_opt:.6f}")
            with col3:
                st.metric("🔄 Iteraciones", len(iteraciones))
            
            # Tabla de iteraciones
            st.markdown("### 📋 Tabla de Iteraciones")
            df = pd.DataFrame(iteraciones)
            
            # Formatear números para mejor visualización
            df_display = df.copy()
            for col in ['a', 'b', 'x1', 'x2', 'f(x1)', 'f(x2)', 'L']:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.6f}")
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Gráfica de la función y proceso de optimización
            st.markdown("### 📈 Visualización del Proceso")
            
            # Crear gráfica
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Gráfica 1: Función completa
            x_range = np.linspace(a_default, b_default, 1000)
            y_range = [evaluar_funcion(x, funcion_seleccionada) for x in x_range]
            
            ax1.plot(x_range, y_range, 'b-', linewidth=2, label='f(x)')
            ax1.axvline(x_opt, color='red', linestyle='--', linewidth=2, label=f'x* = {x_opt:.4f}')
            ax1.scatter([x_opt], [f_opt], color='red', s=100, zorder=5, label=f'f(x*) = {f_opt:.4f}')
            ax1.set_xlabel('x')
            ax1.set_ylabel('f(x)')
            ax1.set_title(f'Función {funcion_seleccionada} - Golden Section Search')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Gráfica 2: Convergencia del intervalo
            longitudes = [iter_data['L'] for iter_data in iteraciones]
            iteraciones_num = [iter_data['Iteración'] for iter_data in iteraciones]
            
            ax2.plot(iteraciones_num, longitudes, 'g-o', linewidth=2, markersize=6)
            ax2.axhline(epsilon, color='red', linestyle='--', alpha=0.7, label=f'Tolerancia = {epsilon}')
            ax2.set_xlabel('Iteración')
            ax2.set_ylabel('Longitud del Intervalo')
            ax2.set_title('Convergencia del Intervalo de Búsqueda')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_yscale('log')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Información adicional sobre el método
            st.markdown("### ℹ️ Información del Método")
            st.info(f"""
            **Golden Section Search Method:**
            - ✅ Convergencia garantizada para funciones unimodales
            - 📏 Ratio de reducción constante: φ ≈ 0.618
            - 🔢 Total de iteraciones: {len(iteraciones)}
            - 📐 Longitud final del intervalo: {iteraciones[-1]['L']:.6f}
            - ⚡ Eficiencia: O(log n) evaluaciones de función
            
            **Características:**
            - La proporción áurea permite la búsqueda más eficiente posible
            - Solo requiere 1 evaluación nueva por iteración (excepto la primera)
            - Convergencia lineal con ratio constante
            """)
            
        except Exception as e:
            st.error(f"❌ Error durante la ejecución: {str(e)}")
            st.info("Verifica que los parámetros estén dentro del dominio de la función.")
    
    # Información teórica del método
    with st.expander("📚 Información Teórica - Golden Section Search"):
        st.markdown("""
        ### 🔍 Golden Section Search Method
        
        El método de búsqueda de la sección áurea es una técnica de optimización unidimensional que utiliza la proporción áurea (φ = (√5-1)/2 ≈ 0.618) para encontrar eficientemente el mínimo de una función unimodal.
        
        #### **Algoritmo:**
        1. **Normalización**: Transformar el intervalo [a,b] a [0,1]
        2. **Inicialización**: Colocar dos puntos usando la proporción áurea
        3. **Iteración**: Eliminar la región con mayor valor de función
        4. **Convergencia**: Continuar hasta que el intervalo sea menor que ε
        
        #### **Ventajas:**
        - ✅ Convergencia garantizada para funciones unimodales
        - ✅ Óptimo en términos de número de evaluaciones
        - ✅ Solo requiere una evaluación nueva por iteración
        - ✅ Convergencia lineal estable
        
        #### **Desventajas:**
        - ❌ Requiere que la función sea unimodal
        - ❌ Convergencia más lenta que métodos de segundo orden
        - ❌ No utiliza información de derivadas
        
        #### **Complejidad:**
        - Temporal: O(log(1/ε))
        - Espacial: O(1)
        """)