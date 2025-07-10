import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import io
import tempfile
import os

def bounding_phase_method(f, x0, delta, max_iterations=100, tolerance=1e-6):
    """
    Implementa el método de fase de acotamiento para encontrar un intervalo
    que contenga el mínimo de una función unimodal.
    
    Parámetros:
    f: función objetivo a minimizar
    x0: punto inicial
    delta: incremento inicial
    max_iterations: número máximo de iteraciones
    tolerance: tolerancia para convergencia
    
    Retorna:
    tuple: (intervalo_inferior, intervalo_superior, historial_puntos, mensajes_log)
    """
    
    # Inicialización
    k = 0
    x_current = x0
    history = [(k, x_current, f(x_current))]
    
    # Para mostrar en Streamlit
    log_messages = []
    log_messages.append(f"Algoritmo de Fase de Acotamiento")
    log_messages.append(f"Punto inicial x^(0) = {x0}")
    log_messages.append(f"Incremento inicial Δ = {delta}")
    log_messages.append(f"f(x^(0)) = {f(x0):.6f}")
    log_messages.append("-" * 50)
        
    # Paso 2: Determinar la dirección
    f_left = f(x0 - abs(delta))
    f_center = f(x0)
    f_right = f(x0 + abs(delta))
    
    log_messages.append(f"Evaluando dirección:")
    log_messages.append(f"f({x0 - abs(delta):.6f}) = {f_left:.6f}")
    log_messages.append(f"f({x0:.6f}) = {f_center:.6f}")
    log_messages.append(f"f({x0 + abs(delta):.6f}) = {f_right:.6f}")
    
    # Determinar si delta es positivo o negativo
    if f_left >= f_center >= f_right:
        delta = abs(delta)  # delta es positivo
        log_messages.append(f"Δ es positivo: {delta}")
    elif f_left <= f_center <= f_right:
        delta = -abs(delta)  # delta es negativo
        log_messages.append(f"Δ es negativo: {delta}")
    else:
        # El mínimo está cerca del punto inicial
        log_messages.append("El mínimo está cerca del punto inicial")
        return (x0 - abs(delta), x0 + abs(delta), history, log_messages)
    
    log_messages.append("-" * 50)
    
    # Pasos 3 y 4: Iteración principal
    x_prev = x_current
    
    for iteration in range(max_iterations):
        # Paso 3: Calcular siguiente punto
        x_next = x_current + (2**k) * delta
        f_next = f(x_next)
        f_current = f(x_current)
        
        log_messages.append(f"Iteración {iteration + 1}:")
        log_messages.append(f"k = {k}")
        log_messages.append(f"x^({k+1}) = x^({k}) + 2^{k} * Δ = {x_current:.6f} + {2**k} * {delta:.6f} = {x_next:.6f}")
        log_messages.append(f"f(x^({k+1})) = f({x_next:.6f}) = {f_next:.6f}")
        log_messages.append(f"f(x^({k})) = f({x_current:.6f}) = {f_current:.6f}")
        
        # Guardar en historial
        history.append((k+1, x_next, f_next))
        
        # Paso 4: Condición de parada
        if f_next < f_current:
            log_messages.append(f"f(x^({k+1})) < f(x^({k})) → Continuar")
            x_prev = x_current
            x_current = x_next
            k += 1
        else:
            log_messages.append(f"f(x^({k+1})) >= f(x^({k})) → Parar")
            break
        
        # Verificar convergencia
        if abs(x_next - x_current) < tolerance:
            log_messages.append("Convergencia alcanzada")
            break
        
        log_messages.append("-" * 30)
    
    # Determinar el intervalo final
    if delta > 0:
        interval_left = x_prev
        interval_right = x_next
    else:
        interval_left = x_next
        interval_right = x_prev
    
    log_messages.append("-" * 50)
    log_messages.append(f"RESULTADO:")
    log_messages.append(f"Intervalo de acotamiento: [{interval_left:.6f}, {interval_right:.6f}]")
    log_messages.append(f"Longitud del intervalo: {abs(interval_right - interval_left):.6f}")
    
    return (interval_left, interval_right, history, log_messages)

def plot_bounding_phase(f, x0, delta, interval, history, x_range=None):
    """
    Grafica la función y el proceso del algoritmo de fase de acotamiento
    """
    if x_range is None:
        x_min = min([h[1] for h in history]) - 2
        x_max = max([h[1] for h in history]) + 2
        x_range = (x_min, x_max)
    
    # Crear puntos para graficar la función
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = [f(xi) for xi in x]
    
    # Crear la gráfica
    plt.figure(figsize=(12, 8))
    
    # Graficar la función
    plt.plot(x, y, 'b-', linewidth=2, label='f(x)')
    
    # Graficar los puntos del algoritmo
    x_points = [h[1] for h in history]
    y_points = [h[2] for h in history]
    
    plt.plot(x_points, y_points, 'ro-', markersize=8, linewidth=2, label='Puntos del algoritmo')

    # Anotar los puntos
    for i, (k, x_val, f_val) in enumerate(history):
        plt.annotate(f'x^({k})', (x_val, f_val), 
                    xytext=(5, 5), textcoords='offset points')
    
    # Marcar el intervalo final
    plt.axvline(x=interval[0], color='g', linestyle='--', alpha=0.7, 
                label=f'Intervalo [{interval[0]:.3f}, {interval[1]:.3f}]')
    plt.axvline(x=interval[1], color='g', linestyle='--', alpha=0.7)
    plt.axvspan(interval[0], interval[1], alpha=0.2, color='green')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Método de Fase de Acotamiento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt

def plot_bounding_phase_animation(f, history, x_range=None):
    """
    Crea una animación mostrando el progreso del algoritmo de fase de acotamiento
    """
    if x_range is None:
        x_min = min([h[1] for h in history]) - 2
        x_max = max([h[1] for h in history]) + 2
        x_range = (x_min, x_max)
    
    # Crear puntos para graficar la función
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = [f(xi) for xi in x]
    
    # Inicializar la figura
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Animación del Método de Fase de Acotamiento')
    ax.grid(True, alpha=0.3)
    
    # Inicializar elementos de la gráfica
    line, = ax.plot([], [], 'ro-', markersize=8, linewidth=2, label='Puntos del algoritmo')
    text = ax.text(0, 0, '', fontsize=10, verticalalignment='top')
    ax.legend()
    
    def init():
        line.set_data([], [])
        text.set_text('')
        return line, text
    
    def update(frame):
        # Mostrar puntos hasta la iteración actual
        x_points = [h[1] for h in history[:frame+1]]
        y_points = [h[2] for h in history[:frame+1]]
        line.set_data(x_points, y_points)
        
        # Actualizar la anotación del último punto
        k, x_val, f_val = history[frame]
        text.set_position((x_val, f_val))
        text.set_text(f'x^({k})')
        
        return line, text
    
    # Crear la animación
    ani = animation.FuncAnimation(fig, update, frames=len(history), 
                                 init_func=init, blit=True, interval=1000)
    
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

def show_fase_acotamiento(FUNCIONES, evaluar_funcion):
    """Función principal que muestra la interfaz del Método de Fase de Acotamiento"""
    st.markdown('<div class="grid-background">Método de Fase de Acotamiento - Método de Optimización</div>', unsafe_allow_html=True)
    
    # Selector de función
    st.markdown("### Seleccionar Función")
    funcion_seleccionada = st.selectbox(
        "Elige la función a optimizar:",
        list(FUNCIONES.keys()),
        index=0,
        key="fase_acotamiento_funcion"
    )
    
    # Mostrar información de la función seleccionada
    st.markdown("### Función Seleccionada:")
    st.latex(FUNCIONES[funcion_seleccionada]["latex"])
    st.markdown(f"**Dominio sugerido:** {FUNCIONES[funcion_seleccionada]['dominio']}")
    
    # Crear columnas para los parámetros
    col1, col2, col3, col4 = st.columns(4)
    
    # Valores por defecto basados en la función seleccionada
    if funcion_seleccionada == "Función 1":
        default_x0 = 1.0
        default_delta = 0.5
    elif funcion_seleccionada == "Función 2":
        default_x0 = 0.5
        default_delta = 0.25
    elif funcion_seleccionada == "Función 3":
        default_x0 = 0.0
        default_delta = 0.5
    elif funcion_seleccionada == "Función 4":
        default_x0 = 0.0
        default_delta = 0.3
    elif funcion_seleccionada == "Función Lata":
        default_x0 = 1.0
        default_delta = 0.5
    elif funcion_seleccionada == "Función Caja":
        default_x0 = 1.0
        default_delta = 0.3
    else:
        default_x0 = 0.0
        default_delta = 0.5
    
    with col1:
        x0 = st.number_input("Punto inicial (x₀)", value=default_x0, step=0.1, key="fase_x0")
    with col2:
        delta = st.number_input("Incremento (Δ)", value=default_delta, step=0.1, key="fase_delta")
    with col3:
        max_iter = st.number_input("Máx. iteraciones", value=100, min_value=1, step=1, key="fase_max_iter")
    with col4:
        tolerance = st.number_input("Tolerancia", value=1e-6, format="%.1e", key="fase_tolerance")
    
    # Botón para ejecutar la búsqueda
    if st.button("🔍 Ejecutar Método de Fase de Acotamiento", type="primary"):
        try:
            # Crear función lambda para el algoritmo
            def f_objetivo(x):
                return evaluar_funcion(x, funcion_seleccionada)
            
            # Ejecutar el algoritmo
            resultado = bounding_phase_method(f_objetivo, x0, delta, max_iter, tolerance)
            interval_left, interval_right, history, log_messages = resultado
            
            # Mostrar la animación primero
            st.markdown("### Animación del Proceso")
            gif_output = plot_bounding_phase_animation(f_objetivo, history)
            st.image(gif_output, caption="Progreso del algoritmo por iteración", use_container_width=True)
            
            # Mostrar la gráfica estática
            st.markdown("### Gráfica Final de la Función")
            fig = plot_bounding_phase(f_objetivo, x0, delta, (interval_left, interval_right), history)
            st.pyplot(fig)
            
            # Mostrar resultados después
            st.markdown("### Resultados del Algoritmo")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Punto Inicial", f"{x0:.6f}")
                st.metric("Incremento Inicial", f"{delta:.6f}")
                st.metric("Tolerancia", f"{tolerance:.1e}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Intervalo de Acotamiento", f"[{interval_left:.6f}, {interval_right:.6f}]")
                st.metric("Longitud del Intervalo", f"{abs(interval_right - interval_left):.6f}")
                st.metric("Iteraciones Realizadas", len(history) - 1)
                st.metric("Puntos Evaluados", len(history))
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Mostrar tabla de iteraciones
            st.markdown("### Historial de Iteraciones")
            if len(history) > 0:
                # Crear tabla con los datos
                import pandas as pd
                df_data = []
                for k, x_val, f_val in history:
                    df_data.append({
                        'k': k,
                        'x^(k)': f"{x_val:.6f}",
                        'f(x^(k))': f"{f_val:.6f}"
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
            
            with st.expander("📋 Log Detallado del Algoritmo"):
                for message in log_messages:
                    st.text(message)
        
        except Exception as e:
            st.error(f"Error durante la ejecución: {str(e)}")
            st.error("Verifica que el punto inicial esté dentro del dominio de la función.")
    
    # Información adicional sobre el algoritmo
    with st.expander("ℹ️ Información sobre el Algoritmo"):
        st.markdown(f"""
        **Método de Fase de Acotamiento para {funcion_seleccionada}:**
        
        Este método encuentra un intervalo que contiene el mínimo de una función unimodal. El algoritmo:
        
        1. **Inicialización**: Establece punto inicial x⁽⁰⁾ y incremento Δ
        2. **Determinación de dirección**: Evalúa f(x⁽⁰⁾-|Δ|), f(x⁽⁰⁾), f(x⁽⁰⁾+|Δ|) para determinar si Δ es positivo o negativo
        3. **Iteración**: Calcula x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ + 2ᵏ·Δ
        4. **Condición de parada**: Si f(x⁽ᵏ⁺¹⁾) ≥ f(x⁽ᵏ⁾), entonces para
        5. **Resultado**: El intervalo [x⁽ᵏ⁻¹⁾, x⁽ᵏ⁺¹⁾] contiene el mínimo
        
        **Ventajas:**
        - Rápida convergencia para funciones unimodales
        - Encuentra un intervalo de acotamiento eficientemente
        - No requiere derivadas
        - Buena preparación para métodos de refinamiento
        
        **Desventajas:**
        - Requiere que la función sea unimodal
        - Sensible al punto inicial y incremento
        - Solo encuentra el intervalo, no el punto exacto
        
        **Función actual:** {funcion_seleccionada}
        **Dominio:** {FUNCIONES[funcion_seleccionada]['dominio']}
        """)