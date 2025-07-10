import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time # Necesario para el retraso en la animación

def busqueda_exhaustiva_core(a, b, n, funcion_nombre, evaluar_funcion):
    """
    Algoritmo de búsqueda exhaustiva para encontrar el mínimo de una función.
    Esta función solo calcula y retorna el historial, no imprime en Streamlit.
    """
    if n <= 0:
        return None, [] # Evitar división por cero si n es 0 o negativo
    
    delta_x = (b - a) / n
    x1 = a
    x2 = x1 + delta_x
    x3 = x2 + delta_x
    
    historial = [] # Lista para guardar los estados de cada iteración
    
    # Bucle principal del algoritmo
    while True:
        # Guardar el estado actual para el historial (antes de la condición de parada)
        # Esto permite visualizar los valores que llevaron a la decisión de parar
        
        # Evaluar funciones, manejando errores con np.inf
        f_x1 = evaluar_funcion(x1, funcion_nombre)
        f_x2 = evaluar_funcion(x2, funcion_nombre)
        f_x3 = evaluar_funcion(x3, funcion_nombre)
        
        historial.append({
            'x1': x1, 'x2': x2, 'x3': x3,
            'f(x1)': f_x1, 'f(x2)': f_x2, 'f(x3)': f_x3,
            'condicion_cumplida': False, # Se actualiza si se cumple la condición
            'intervalo_actual': (x1, x3)
        })

        # Paso 2: Verificar si el mínimo está en (x1, x3)
        # Se asegura que f_x1, f_x2, f_x3 no sean NaN para la comparación
        if (not np.isnan(f_x1) and not np.isnan(f_x2) and not np.isnan(f_x3)) and \
           (f_x1 >= f_x2 <= f_x3):
            historial[-1]['condicion_cumplida'] = True # Marcar la última iteración como la de parada
            return (x1, x3), historial
        else:
            x1 = x2
            x2 = x3
            x3 = x2 + delta_x
            
            # Paso 3: Verificar si x3 excede b
            if x3 > b or np.isclose(x3, b): # Usar np.isclose para comparación de floats
                # Si x3 excede el límite, verificar los extremos del intervalo original
                f_a = evaluar_funcion(a, funcion_nombre)
                f_b = evaluar_funcion(b, funcion_nombre)

                # Comprobar si los valores son válidos antes de la comparación
                if not np.isinf(f_a) and not np.isnan(f_a) and \
                   (np.isinf(f_b) or np.isnan(f_b) or f_a <= f_b):
                    # Añadir una última iteración si 'a' es el mínimo
                    if not historial[-1]['condicion_cumplida']: # Si no se cumplió la condición antes
                        historial.append({
                            'x1': a, 'x2': a, 'x3': a,
                            'f(x1)': f_a, 'f(x2)': f_a, 'f(x3)': f_a, # Valores repetidos para simular punto
                            'condicion_cumplida': True,
                            'intervalo_actual': (a, a)
                        })
                    return a, historial
                elif not np.isinf(f_b) and not np.isnan(f_b):
                    # Añadir una última iteración si 'b' es el mínimo
                    if not historial[-1]['condicion_cumplida']: # Si no se cumplió la condición antes
                        historial.append({
                            'x1': b, 'x2': b, 'x3': b,
                            'f(x1)': f_b, 'f(x2)': f_b, 'f(x3)': f_b, # Valores repetidos para simular punto
                            'condicion_cumplida': True,
                            'intervalo_actual': (b, b)
                        })
                    return b, historial
                else: # Si ni a ni b son válidos o algo salió mal
                    return (a + b) / 2, historial # Retornar el centro como un intento
        
        # Limitar el número de iteraciones en el core para evitar bucles infinitos
        if len(historial) > 2000: # Un límite razonable para evitar cuelgues
            return (a + b) / 2, historial # Retornar un valor por defecto si excede las iteraciones

def graficar_funcion_animacion(a, b, funcion_nombre, evaluar_funcion, current_x1, current_x2, current_x3, result_interval=None, min_point=None, ax=None):
    """Grafica la función en el intervalo dado, con marcadores para la animación."""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        ax.clear() # Limpiar el eje si ya existe para redibujar

    x_vals = []
    y_vals = []
    
    # Rango de la gráfica: se extiende un poco más allá del intervalo original para contexto
    plot_a = a - (b - a) * 0.1
    plot_b = b + (b - a) * 0.1
    
    # Asegúrate de que el rango de ploteo no vaya por debajo de 0 si la función no está definida ahí
    if funcion_nombre in ["Función 1", "Función Lata"] and plot_a < 0:
        plot_a = 0.001 # Asegurar que no se intente graficar en 0 o negativo

    x_range = np.linspace(plot_a, plot_b, 500)
    
    for x in x_range:
        y = evaluar_funcion(x, funcion_nombre)
        # Solo añade puntos si el valor es un número finito
        if not np.isinf(y) and not np.isnan(y):
            x_vals.append(x)
            y_vals.append(y)
    
    if x_vals and y_vals: # Solo plotear si hay datos válidos
        ax.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'{funcion_nombre}')
    
    # Marcar los puntos x1, x2, x3 actuales
    if current_x1 is not None:
        try:
            f1 = evaluar_funcion(current_x1, funcion_nombre)
            f2 = evaluar_funcion(current_x2, funcion_nombre)
            f3 = evaluar_funcion(current_x3, funcion_nombre)

            if not np.isinf(f1) and not np.isnan(f1):
                ax.plot(current_x1, f1, 'go', markersize=8, label=r'$x_1$')
                ax.axvline(current_x1, color='g', linestyle=':', alpha=0.6)
            if not np.isinf(f2) and not np.isnan(f2):
                ax.plot(current_x2, f2, 'yo', markersize=8, label=r'$x_2$ (Mínimo Candidato)')
                ax.axvline(current_x2, color='y', linestyle=':', alpha=0.6)
            if not np.isinf(f3) and not np.isnan(f3):
                ax.plot(current_x3, f3, 'co', markersize=8, label=r'$x_3$')
                ax.axvline(current_x3, color='c', linestyle=':', alpha=0.6)
        except Exception:
            pass # No graficar si la evaluación falla
    
    # Marcar el intervalo final si se ha encontrado
    if result_interval is not None:
        x_min, x_max = result_interval
        ax.axvspan(x_min, x_max, alpha=0.3, color='red',
                   label=f'Intervalo Final: [{x_min:.4f}, {x_max:.4f}]')
    elif min_point is not None:
        try:
            y_min_point = evaluar_funcion(min_point, funcion_nombre)
            if not np.isinf(y_min_point) and not np.isnan(y_min_point):
                ax.plot(min_point, y_min_point, 'ro', markersize=10, label=f'Mínimo Estimado: {min_point:.4f}')
        except Exception:
            pass

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Búsqueda Exhaustiva: {funcion_nombre}')
    ax.legend()
    ax.set_xlim(plot_a, plot_b) # Asegura que el eje x se ajuste al rango de ploteo
    
    # Calcular y_limits basados en los valores de la función dentro del rango visible
    # Filtra np.inf y np.nan para calcular límites válidos
    valid_y_vals = [y for y in y_vals if not np.isinf(y) and not np.isnan(y)]
    if valid_y_vals:
        y_min_plot = np.min(valid_y_vals)
        y_max_plot = np.max(valid_y_vals)
        # Ajustar los límites del eje y con un pequeño margen
        margin = (y_max_plot - y_min_plot) * 0.1
        ax.set_ylim(y_min_plot - margin, y_max_plot + margin)
    else:
        ax.set_ylim(-10, 10) # Fallback si no hay valores válidos para plotear

    return plt.gcf() # Retorna la figura actual

def show_busqueda_exhaustiva(FUNCIONES, evaluar_funcion):
    """Función principal que muestra la interfaz de Búsqueda Exhaustiva con animación."""
    st.markdown(f'<h1 class="main-title">Búsqueda Exhaustiva</h1>', unsafe_allow_html=True)
    st.markdown("""
    El método de **Búsqueda Exhaustiva** (o Directa) es un algoritmo de optimización univariado que busca el mínimo de una función evaluando sistemáticamente puntos en un intervalo dado. Es simple pero puede ser computacionalmente costoso para alta precisión.
    """)
    
    st.subheader("Configuración del Algoritmo")
    
    col1, col2 = st.columns(2)

    with col1:
        funcion_seleccionada = st.selectbox(
            "Elige la función a optimizar:",
            list(FUNCIONES.keys()),
            key="bs_func_select" # Clave única para evitar conflictos
        )
        st.latex(FUNCIONES[funcion_seleccionada]["latex"])
        st.info(f"Dominio sugerido: {FUNCIONES[funcion_seleccionada]['dominio']}")

    # Valores por defecto basados en la función seleccionada para los inputs
    # ¡CORRECCIÓN APLICADA AQUÍ!
    default_a = FUNCIONES[funcion_seleccionada]["intervalos"][0][0]
    default_b = FUNCIONES[funcion_seleccionada]["intervalos"][0][1]

    with col2:
        a = st.number_input("Límite inferior (a):", value=float(default_a), step=0.1, key="bs_a")
        b = st.number_input("Límite superior (b):", value=float(default_b), step=0.1, key="bs_b")
        n = st.number_input("Número de puntos (n):", value=100, min_value=10, max_value=2000, step=10, key="bs_n")
        
        anim_speed = st.slider("Velocidad de la animación (segundos por paso):", min_value=0.01, max_value=2.0, value=0.2, step=0.05)

    if st.button("🚀 Ejecutar Búsqueda Exhaustiva con Animación", type="primary"):
        if a >= b:
            st.error("El límite inferior 'a' debe ser menor que el límite superior 'b'.")
            return
        if n <= 0:
            st.error("El número de puntos 'n' debe ser mayor que 0.")
            return

        st.subheader("Proceso de Búsqueda (Animación)")
        
        # Crear placeholders para la información y el gráfico
        info_placeholder = st.empty()
        graph_placeholder = st.empty()
        
        # Inicializar la figura y el eje para el gráfico de animación
        fig, ax = plt.subplots(figsize=(10, 6))
        
        delta_x = (b - a) / n
        x1_current = a
        x2_current = x1_current + delta_x
        x3_current = x2_current + delta_x
        
        iteration_count = 0
        final_result = None

        # Función auxiliar para formatear valores que podrían ser infinitos para LaTeX
        def format_val_latex(val):
            if np.isinf(val):
                return "\\infty" # Símbolo de infinito en LaTeX
            elif np.isnan(val):
                return "NaN"
            else:
                return f"{val:.4f}"

        # Bucle de animación (replica la lógica de busqueda_exhaustiva_core)
        while True:
            iteration_count += 1
            
            # Evaluar funciones, manejando errores con np.inf
            f_x1 = evaluar_funcion(x1_current, funcion_seleccionada)
            f_x2 = evaluar_funcion(x2_current, funcion_seleccionada)
            f_x3 = evaluar_funcion(x3_current, funcion_seleccionada)
            
            # Actualizar placeholder de información
            with info_placeholder.container():
                st.markdown(f"**Iteración:** `{iteration_count}`")
                st.write(f"**Puntos actuales:**")
                st.latex(f"x_1 = {x1_current:.4f}, x_2 = {x2_current:.4f}, x_3 = {x3_current:.4f}")
                st.write(f"**Valores de la función:**")
                # ¡CORRECCIÓN APLICADA AQUÍ!
                st.latex(f"f(x_1) = {format_val_latex(f_x1)}, f(x_2) = {format_val_latex(f_x2)}, f(x_3) = {format_val_latex(f_x3)}")
                
            # Actualizar placeholder del gráfico
            with graph_placeholder:
                current_fig = graficar_funcion_animacion(a, b, funcion_seleccionada, evaluar_funcion,
                                                         x1_current, x2_current, x3_current,
                                                         ax=ax)
                st.pyplot(current_fig)
            
            time.sleep(anim_speed) # Pausa para la animación
            
            # Verificar la condición de parada
            if (not np.isnan(f_x1) and not np.isnan(f_x2) and not np.isnan(f_x3)) and \
               (f_x1 >= f_x2 <= f_x3):
                final_result = (x1_current, x3_current)
                st.success(f"**Mínimo encontrado en el intervalo:** `[{x1_current:.4f}, {x3_current:.4f}]`")
                break
            else:
                x1_current = x2_current
                x2_current = x3_current
                x3_current = x2_current + delta_x
                
                if x3_current > b or np.isclose(x3_current, b): # Usar np.isclose
                    st.warning("Se alcanzó el límite superior 'b'. Verificando extremos finales.")
                    
                    f_a = evaluar_funcion(a, funcion_seleccionada)
                    f_b = evaluar_funcion(b, funcion_seleccionada)

                    if not np.isinf(f_a) and not np.isnan(f_a) and \
                       (np.isinf(f_b) or np.isnan(f_b) or f_a <= f_b):
                        final_result = a
                        st.success(f"**Mínimo final encontrado en el punto:** `{a:.4f}`")
                    elif not np.isinf(f_b) and not np.isnan(f_b):
                        final_result = b
                        st.success(f"**Mínimo final encontrado en el punto:** `{b:.4f}`")
                    else:
                        st.error("No se pudo determinar un mínimo claro en los límites del intervalo.")
                        final_result = (a + b) / 2 # Un valor por defecto si no se encuentra
                    break
            
            # Esto es un resguardo si la función es compleja y las evaluaciones fallan repetidamente
            if iteration_count > 2000: # Limitar para evitar bucles infinitos en casos extremos
                st.error("Demasiadas iteraciones sin convergencia. Deteniendo la animación.")
                final_result = (a + b) / 2 # Un valor por defecto si no converge
                break

        # Una vez que la animación termina, mostrar los resultados finales estáticamente
        st.subheader("Resultados Finales")
        
        # Llamar a la función core para obtener el historial completo y métricas
        resultado_core, iteraciones_completas = busqueda_exhaustiva_core(a, b, n, funcion_seleccionada, evaluar_funcion)

        precision = 2 * (b - a) / n
        
        col1_res, col2_res = st.columns(2)
        
        with col1_res:
            st.metric("Precisión", f"{precision:.6f}")
            if isinstance(resultado_core, tuple):
                st.metric("Intervalo del Mínimo", f"[{resultado_core[0]:.4f}, {resultado_core[1]:.4f}]")
                x_centro = (resultado_core[0] + resultado_core[1]) / 2
                
                # Manejo para f(centro)
                try:
                    f_centro = evaluar_funcion(x_centro, funcion_seleccionada)
                    st.metric("f(centro)", f"{format_val_latex(f_centro)}")
                except Exception:
                    st.metric("f(centro)", "No evaluable")
            else:
                st.metric("Punto Mínimo", f"{resultado_core:.4f}")
                # Manejo para f(mínimo)
                try:
                    f_minimo = evaluar_funcion(resultado_core, funcion_seleccionada)
                    st.metric("f(mínimo)", f"{format_val_latex(f_minimo)}")
                except Exception:
                    st.metric("f(mínimo)", "No evaluable")
        
        with col2_res:
            st.metric("Iteraciones Realizadas", len(iteraciones_completas))
            st.metric("Intervalo Inicial", f"[{a}, {b}]")
            st.metric("Puntos Evaluados (n)", n)
            st.metric("Función Utilizada", funcion_seleccionada)
        
        # Gráfico final con el resultado marcado
        st.markdown("### Gráfica Final del Mínimo")
        final_fig = None
        if isinstance(final_result, tuple):
            final_fig = graficar_funcion_animacion(a, b, funcion_seleccionada, evaluar_funcion, None, None, None, result_interval=final_result)
        else:
            final_fig = graficar_funcion_animacion(a, b, funcion_seleccionada, evaluar_funcion, None, None, None, min_point=final_result)
        st.pyplot(final_fig)
        
        st.markdown("### Detalles de las Iteraciones (Historial Completo)")
        if len(iteraciones_completas) > 0:
            # Mostrar todas las iteraciones en un expander
            with st.expander("Ver todas las iteraciones paso a paso"):
                for i, iter_data in enumerate(iteraciones_completas):
                    st.markdown(f"**Iteración {i+1}:**")
                    col_det1, col_det2, col_det3 = st.columns(3)
                    with col_det1:
                        st.write(f"x₁ = {iter_data['x1']:.4f}")
                        st.write(f"f(x₁) = {format_val_latex(iter_data['f(x1)'])}")
                    with col_det2:
                        st.write(f"x₂ = {iter_data['x2']:.4f}")
                        st.write(f"f(x₂) = {format_val_latex(iter_data['f(x2)'])}")
                    with col_det3:
                        st.write(f"x₃ = {iter_data['x3']:.4f}")
                        st.write(f"f(x₃) = {format_val_latex(iter_data['f(x3)'])}")
                    
                    if iter_data['condicion_cumplida']:
                        st.success("✓ Condición de parada cumplida en esta iteración.")
                    st.markdown("---") # Separador para cada iteración
        else:
            st.info("No hay historial de iteraciones para mostrar.")
            
    with st.expander("ℹ️ Información sobre el Algoritmo"):
        st.markdown(f"""
        **Búsqueda Exhaustiva para {funcion_seleccionada}:**
        
        Este método de optimización busca el mínimo de una función evaluando sistemáticamente
        puntos en un intervalo dado. El algoritmo:
        
        1. **Divide** el intervalo `[a,b]` en `n` subintervalos.
        2. **Evalúa** tres puntos consecutivos: $x_1$, $x_2$, $x_3$.
        3. **Verifica** si se cumple la condición: $f(x_1) \geq f(x_2) \leq f(x_3)$.
        4. Si se cumple, el mínimo está en $[x_1, x_3]$.
        5. Si no, **avanza** un paso ($x_1 \leftarrow x_2$, $x_2 \leftarrow x_3$, $x_3 \leftarrow x_2 + \Delta x$) y repite.
        
        **Ventajas:**
        - Garantiza encontrar el mínimo global en el intervalo (si `n` es suficientemente grande).
        - Simple de implementar y entender.
        - No requiere derivadas.
        - Funciona con cualquier función continua.
        
        **Desventajas:**
        - Computacionalmente costoso para alta precisión (requiere un `n` muy grande).
        - El tiempo de ejecución aumenta linealmente con `n`.
        
        **Función actual:** {funcion_seleccionada}
        **Dominio:** {FUNCIONES[funcion_seleccionada]['dominio']}
        """)