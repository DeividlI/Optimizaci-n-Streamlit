import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==============================================================================
# LÓGICA MATEMÁTICA DEL ALGORITMO
# ==============================================================================

def movimiento_exploratorio(x_base, delta, funcion):
    """
    Realiza una búsqueda exploratoria alrededor de un punto base.
    Devuelve un nuevo punto si se encuentra una mejora, y un booleano de éxito.
    """
    N = len(x_base)
    x_actual = np.copy(x_base)
    
    # Puntos explorados en esta búsqueda (para visualización)
    puntos_explorados = [np.copy(x_base)]

    for i in range(N):
        f_actual = funcion(x_actual)
        
        # Probar en dirección positiva
        x_plus = np.copy(x_actual)
        x_plus[i] += delta[i]
        f_plus = funcion(x_plus)
        puntos_explorados.append(x_plus)
        
        if f_plus < f_actual:
            x_actual = x_plus
            continue

        # Probar en dirección negativa
        x_minus = np.copy(x_actual)
        x_minus[i] -= delta[i]
        f_minus = funcion(x_minus)
        puntos_explorados.append(x_minus)
        
        if f_minus < f_actual:
            x_actual = x_minus

    # Comprobar si hubo una mejora respecto al punto de partida
    if funcion(x_actual) < funcion(x_base):
        return x_actual, True, puntos_explorados
    else:
        return x_base, False, puntos_explorados

def movimiento_patron(x_arreglo):
    """Calcula el siguiente punto de patrón."""
    # x_arreglo[k] + (x_arreglo[k] - x_arreglo[k-1])
    return x_arreglo[-1] + (x_arreglo[-1] - x_arreglo[-2])

# ==============================================================================
# FUNCIÓN PRINCIPAL DEL ALGORITMO 
# ==============================================================================

def hooke_jeeves_algorithm(objective_func, x0, delta, alpha, epsilon, max_iter):
    """
    Ejecuta el algoritmo de Hooke-Jeeves.
    Devuelve el historial de puntos y los datos para la tabla de iteraciones.
    """
    k = 0
    x_base_hist = [np.copy(x0)] # Historial de puntos base (tu x_arreglo)
    
    # Historial detallado para visualización
    history = {'base': [np.copy(x0)], 'pattern': [], 'exploratory_paths': []}
    table_data = []
    
    iter_count = 0
    while np.linalg.norm(delta) > epsilon and iter_count < max_iter:
        iter_count += 1
        
        # 1. Movimiento Exploratorio desde el último punto base
        x_nuevo, exito, explorados = movimiento_exploratorio(x_base_hist[-1], delta, objective_func)
        history['exploratory_paths'].append(explorados)
        
        if exito:
            # Si el movimiento exploratorio tuvo éxito, se establece un nuevo punto base
            x_base_hist.append(x_nuevo)
            history['base'].append(x_nuevo)
            
            # 2. Movimiento de Patrón
            xp = movimiento_patron(x_base_hist)
            history['pattern'].append(xp)
            
            # 3. Segundo Movimiento Exploratorio desde el punto de patrón
            x_final, exito_final, explorados_final = movimiento_exploratorio(xp, delta, objective_func)
            history['exploratory_paths'].append(explorados_final)
            
            # Si el segundo movimiento fue exitoso, se convierte en el nuevo punto base
            if exito_final:
                x_base_hist.append(x_final)
                history['base'].append(x_final)
            
            operation = "Éxito -> Patrón -> Exploración"
        else:
            # Si el movimiento exploratorio falla, se reduce el paso
            delta = delta / alpha
            operation = "Fallo -> Reducción de delta"

        table_data.append({
            'Iteración': iter_count,
            'Punto Base': f"[{x_base_hist[-1][0]:.3f}, {x_base_hist[-1][1]:.3f}]",
            'f(Punto Base)': objective_func(x_base_hist[-1]),
            '|delta|': np.linalg.norm(delta),
            'Operación': operation
        })

    x_best = x_base_hist[-1]
    f_best = objective_func(x_best)
    return x_best, f_best, history, table_data

# ==============================================================================
# INTERFAZ DE STREAMLIT
# ==============================================================================

def plot_contour_with_hj_path(ax, func, bounds, history):
    """Dibuja el mapa de contorno y la ruta completa de Hooke-Jeeves."""
    min_b, max_b = [b[0] for b in bounds], [b[1] for b in bounds]
    x_vals = np.linspace(min_b[0], max_b[0], 100)
    y_vals = np.linspace(min_b[1], max_b[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.array([func(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

    ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
    
    # Dibuja la ruta
    base_points = np.array(history['base'])
    pattern_points = np.array(history['pattern'])
    
    # Línea principal que conecta los puntos base
    ax.plot(base_points[:, 0], base_points[:, 1], 'w-', label='Ruta de Puntos Base', markersize=5, zorder=3)
    ax.scatter(base_points[:, 0], base_points[:, 1], c='cyan', edgecolor='k', marker='o', s=50, label='Puntos Base', zorder=4)
    
    if len(pattern_points) > 0:
        ax.scatter(pattern_points[:, 0], pattern_points[:, 1], c='magenta', marker='P', s=80, label='Movimientos de Patrón', zorder=5)

    # Dibuja los pequeños movimientos exploratorios
    for path in history['exploratory_paths']:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], 'r--', linewidth=0.8, alpha=0.7, zorder=2)

    ax.scatter(base_points[0, 0], base_points[0, 1], c='lime', edgecolor='k', marker='o', s=80, label='Inicio', zorder=5)
    ax.scatter(base_points[-1, 0], base_points[-1, 1], c='gold', edgecolor='k', marker='*', s=150, label='Mejor Solución', zorder=5)
    
    ax.set_title("Ruta de Optimización de Hooke-Jeeves")
    ax.set_xlabel('x_0')
    ax.set_ylabel('x_1')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

def show_hooke_jeeves(FUNCIONES, evaluar_funcion):
    st.markdown("## 🎯 Hooke-Jeeves - Búsqueda Directa con Patrones")
    st.markdown("""
    Este método de búsqueda directa no requiere derivadas. Combina dos tipos de movimientos para encontrar el mínimo de una función:
    - **Movimiento Exploratorio**: Realiza una búsqueda local alrededor del punto base actual, moviéndose a lo largo de cada eje para encontrar una dirección de mejora.
    - **Movimiento de Patrón**: Si la búsqueda exploratoria tuvo éxito, el algoritmo "acelera" en la dirección de la mejora encontrada, saltando a un nuevo punto. Desde este nuevo punto, se realiza otro movimiento exploratorio para refinar la posición.
    
    Si un movimiento exploratorio falla, el tamaño del paso (`delta`) se reduce y la búsqueda se vuelve más fina.
    """)

    with st.expander("Configuración del Algoritmo y la Función", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            funcion_seleccionada = st.selectbox("🎯 Selecciona la función:", list(FUNCIONES.keys()), key="hj_func")
            info_funcion = FUNCIONES[funcion_seleccionada]
            
            st.markdown("**Parámetros de Ejecución:**")
            alpha = st.slider("α (Factor de reducción de delta)", 1.1, 4.0, 2.0, 0.1)
            max_iter = st.slider("🔄 Máximo de iteraciones:", 10, 200, 50, 10)
            epsilon = st.number_input("ε (Tolerancia de |delta|)", 0.0, 0.1, 0.001, format="%.5f")
        with col2:
            st.latex(info_funcion['latex'])
            st.markdown("**📍 Punto y Paso Iniciales:**")
            bounds_list = info_funcion["intervalos"]
            
            x0_cols = st.columns(len(bounds_list))
            x0_vals = [c.number_input(f'x0_{i}', value=0.0, key=f'hj_x0_{i}') for i, c in enumerate(x0_cols)]
            
            delta_cols = st.columns(len(bounds_list))
            delta_vals = [c.number_input(f'delta0_{i}', value=0.5, key=f'hj_d0_{i}') for i, c in enumerate(delta_cols)]
            
            x0 = np.array(x0_vals)
            delta0 = np.array(delta_vals)

    if st.button("🚀 Ejecutar Hooke-Jeeves", key="run_hj"):
        func_objetivo = lambda x: evaluar_funcion(x, funcion_seleccionada)
        
        with st.spinner("Optimizando con Hooke-Jeeves..."):
            x_best, f_best, history, table_data = hooke_jeeves_algorithm(
                func_objetivo, x0, delta0, alpha, epsilon, max_iter
            )
        
        st.success("✅ Optimización completada!")
        
        col1, col2 = st.columns(2)
        col1.metric("🎯 Mejor solución (x_best)", f"[{x_best[0]:.4f}, {x_best[1]:.4f}]")
        col2.metric("📊 Valor óptimo f(x_best)", f"{f_best:.6f}")

        st.markdown("### 📈 Visualización de la Optimización")
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_contour_with_hj_path(ax, func_objetivo, info_funcion["intervalos"], history)
        st.pyplot(fig)
        
        st.markdown("### 📋 Tabla de Iteraciones")
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        # --- Contexto y Aplicaciones ---
        st.markdown("---")
        st.subheader("Contexto y Aplicaciones")
        with st.expander("Acerca del Algoritmo Hooke-Jeeves"):
            st.markdown("""
            El **algoritmo Hooke-Jeeves** es un método de búsqueda directa que no requiere el cálculo de derivadas para encontrar el mínimo de una función objetivo en un espacio multidimensional. Combina **movimientos exploratorios**, que buscan mejoras locales a lo largo de los ejes coordenados, con **movimientos de patrón**, que aprovechan las mejoras previas para realizar saltos más grandes en la dirección de descenso. Este enfoque permite al algoritmo explorar el espacio de manera eficiente y adaptarse mediante la reducción del tamaño del paso (\(\delta\)) cuando no se encuentran mejoras.

            ### ¿Cómo funciona?
            - **Inicialización**: Se parte de un punto inicial \(x_0\) y un vector de pasos iniciales \(\delta\) para cada dimensión.
            - **Movimiento Exploratorio**: Evalúa la función objetivo en puntos vecinos a lo largo de cada eje (positiva y negativamente) para encontrar una mejora. Si se encuentra un punto mejor, se convierte en el nuevo punto base.
            - **Movimiento de Patrón**: Si el movimiento exploratorio tiene éxito, el algoritmo calcula un nuevo punto extrapolando la dirección de mejora (\(x_k + (x_k - x_{k-1})\)) y realiza otra búsqueda exploratoria desde este punto.
            - **Reducción del Paso**: Si el movimiento exploratorio falla, el tamaño del paso \(\delta\) se reduce por un factor \(\alpha\), y se repite la búsqueda exploratoria.
            - **Criterio de Parada**: El algoritmo termina cuando la norma de \(\delta\) es menor que una tolerancia \(\epsilon\) o se alcanza el número máximo de iteraciones.

            ### Aplicaciones
            - **Optimización no lineal**: Ideal para funciones donde las derivadas no están disponibles o son costosas de calcular.
            - **Aprendizaje automático**: Utilizado para optimizar hiperparámetros o funciones de pérdida en modelos donde los gradientes no son accesibles.
            - **Ingeniería**: Aplicado en el diseño de sistemas, ajuste de parámetros en modelos físicos o simulaciones (por ejemplo, optimización de estructuras o procesos químicos).
            - **Ciencias computacionales**: Útil en problemas de baja a moderada dimensión donde se requiere una solución robusta y simple.

            ### Ventajas
            - No depende de derivadas, lo que lo hace adecuado para funciones no diferenciables, discontinuas o ruidosas.
            - Relativamente simple de implementar y entender.
            - El uso de movimientos de patrón permite acelerar la convergencia en direcciones prometedoras.

            ### Limitaciones
            - Puede quedarse atrapado en mínimos locales, especialmente en funciones con múltiples óptimos.
            - La convergencia puede ser lenta si el tamaño inicial del paso (\(\delta\)) no es adecuado.
            - Menos eficiente en espacios de alta dimensión en comparación con métodos basados en gradientes.
            - La elección del punto inicial y los pasos iniciales afecta significativamente el rendimiento.

            Este código proporciona una implementación educativa del algoritmo Hooke-Jeeves, con visualizaciones que muestran la trayectoria de los puntos base, los movimientos de patrón y las búsquedas exploratorias en un espacio 2D, facilitando la comprensión de cómo el algoritmo navega hacia el óptimo.
            """)