# cauchy_method.py

import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

def regla_eliminacion(x1, x2, fx1, fx2, a, b):
    if fx1 > fx2:
        return x1, b
    if fx1 < fx2:
        return a, x2
    return x1, x2

def w_to_x(w: float, a, b):
    return w * (b - a) + a

def busquedaDorada(funcion, epsilon: float, a: float = 0.0, b: float = 1.0):
    """Búsqueda de la Sección Dorada para la optimización de línea."""
    PHI = (1 + math.sqrt(5)) / 2 - 1
    aw, bw = 0, 1
    Lw = 1
    for _ in range(100): 
        if Lw < epsilon:
            break
        w1 = aw + PHI * Lw
        w2 = bw - PHI * Lw
        aw, bw = regla_eliminacion(w1, w2, funcion(w_to_x(w1, a, b)), 
                                   funcion(w_to_x(w2, a, b)), aw, bw)
        Lw = bw - aw
    return (w_to_x(aw, a, b) + w_to_x(bw, a, b)) / 2

def gradiente(f, x, deltaX=0.001):
    """Calcula el gradiente numérico de una función en el punto x."""
    grad = []
    for i in range(len(x)):
        xp = np.copy(x) # Usar np.copy() para evitar modificar el original
        xp[i] = xp[i] + deltaX
        xn = np.copy(x) # Usar np.copy() para evitar modificar el original
        xn[i] = xn[i] - deltaX
        grad.append((f(xp) - f(xn)) / (2 * deltaX))
    return np.array(grad)

# --- Versión adaptada del algoritmo para guardar el historial ---
def cauchy_algorithm(funcion, x0, epsilon1, epsilon2, M):
    """
    Tu implementación del método de Cauchy, adaptada para registrar el historial.
    """
    xk = np.copy(x0)
    k = 0
    
    # Historial para visualización y análisis
    path_history = [np.copy(xk)]
    table_data = []

    while k < M:
        grad = gradiente(funcion, xk)
        grad_norm = np.linalg.norm(grad)
        
        # Criterio de paro 1: Norma del gradiente muy pequeña
        if grad_norm < epsilon1:
            break
            
        def alpha_funcion(alpha):
            return funcion(xk - alpha * grad)
        
        # Búsqueda de línea para encontrar el alpha óptimo
        alpha = busquedaDorada(alpha_funcion, epsilon2, a=0.0, b=1.0)
        
        x_k1 = xk - alpha * grad
        
        # Criterio de paro 2: Cambio relativo muy pequeño
        if np.linalg.norm(x_k1 - xk) / (np.linalg.norm(xk) + 1e-10) < epsilon2 and k > 0: # Añadir 1e-10 para evitar división por cero
            break

        # Guardar historial después de aplicar los criterios de parada para asegurar que el último punto es válido
        path_history.append(np.copy(x_k1))
        table_data.append({
            'Iteración (k)': k,
            'xk': f"[{', '.join([f'{val:.3f}' for val in xk])}]", # Formato dinámico para xk
            'f(xk)': funcion(xk),
            '|grad(f(xk))|': grad_norm,
            'alpha*': alpha,
        })
        
        k += 1
        xk = x_k1
        
    return xk, path_history, table_data


def plot_cauchy_results(func, bounds, path_history, table_data):
    """
    Dibuja los resultados del método de Cauchy como figuras separadas.
    Retorna una lista de tuplas (titulo, figura).
    """
    figures = []

    # --- Figure 1: Mapa de Contorno y Ruta (solo para funciones 2D) ---
    if len(bounds) == 2:
        fig_contour = plt.figure(figsize=(8, 7))
        ax_contour = fig_contour.add_subplot(111)

        # Determinar rangos para la gráfica basados en los límites del dominio
        x_min_domain, x_max_domain = bounds[0]
        y_min_domain, y_max_domain = bounds[1]

        # Ajustar rangos para incluir la trayectoria completa si es necesario
        path_points = np.array(path_history)
        
        # Si path_points está vacío o tiene solo un punto, usar los límites del dominio
        if path_points.shape[0] > 0:
            x_min_plot = min(x_min_domain, np.min(path_points[:, 0])) - 0.1 * (x_max_domain - x_min_domain)
            x_max_plot = max(x_max_domain, np.max(path_points[:, 0])) + 0.1 * (x_max_domain - x_min_domain)
            y_min_plot = min(y_min_domain, np.min(path_points[:, 1])) - 0.1 * (y_max_domain - y_min_domain)
            y_max_plot = max(y_max_domain, np.max(path_points[:, 1])) + 0.1 * (y_max_domain - y_min_domain)
        else:
            x_min_plot, x_max_plot = x_min_domain, x_max_domain
            y_min_plot, y_max_plot = y_min_domain, y_max_domain


        x_vals = np.linspace(x_min_plot, x_max_plot, 100)
        y_vals = np.linspace(y_min_plot, y_max_plot, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Calcular Z. Asegurarse de que la función pueda manejar entradas NumPy array
        # y que el reshape funcione correctamente para funciones de 2 variables.
        Z = np.array([func(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        ax_contour.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
        ax_contour.contour(X, Y, Z, levels=20, colors='grey', linewidths=0.5, alpha=0.5) # Añadir líneas de contorno
        
        ax_contour.plot(path_points[:, 0], path_points[:, 1], 'r-o', markersize=4, label='Ruta de Descenso')
        ax_contour.scatter(path_points[0, 0], path_points[0, 1], c='lime', s=100, marker='o', edgecolor='k', label='Inicio', zorder=5)
        ax_contour.scatter(path_points[-1, 0], path_points[-1, 1], c='gold', s=150, marker='*', edgecolor='k', label='Final', zorder=5)
        
        ax_contour.set_title("Ruta de Optimización (Descenso de Gradiente)")
        ax_contour.set_xlabel('x_0')
        ax_contour.set_ylabel('x_1')
        ax_contour.legend()
        ax_contour.grid(True, linestyle='--', alpha=0.5)
        fig_contour.tight_layout()
        figures.append(("Mapa de Contorno y Ruta", fig_contour))
    else:
        # Mensaje placeholder si no es una función 2D
        fig_placeholder_contour = plt.figure(figsize=(10, 4))
        ax_placeholder_contour = fig_placeholder_contour.add_subplot(111)
        ax_placeholder_contour.text(0.5, 0.5, 'La visualización de contorno solo está disponible\n para funciones con 2 variables en el dominio.',
                            horizontalalignment='center', verticalalignment='center', transform=ax_placeholder_contour.transAxes, fontsize=12, color='white')
        ax_placeholder_contour.axis('off')
        fig_placeholder_contour.tight_layout()
        figures.append(("Mapa de Contorno (Nota)", fig_placeholder_contour))


    # --- Figure 2: Convergencia de la Norma del Gradiente ---
    fig_convergence = plt.figure(figsize=(10, 6))
    ax_convergence = fig_convergence.add_subplot(111)
    
    grad_norms = [item['|grad(f(xk))|'] for item in table_data]
    ax_convergence.plot(grad_norms, 'b-', linewidth=2)
    ax_convergence.set_title("Convergencia de la Norma del Gradiente")
    ax_convergence.set_xlabel("Iteraciones")
    ax_convergence.set_ylabel("|grad(f(xk))|")
    ax_convergence.set_yscale('log') # Escala logarítmica para mejor visualización
    ax_convergence.grid(True, linestyle='--', alpha=0.5)
    fig_convergence.tight_layout()
    figures.append(("Convergencia de la Norma del Gradiente", fig_convergence))

    return figures

def show_cauchy_method(FUNCIONES, evaluar_funcion):
    st.markdown("## 🌡️ Método de Cauchy (Descenso más Pronunciado)")
    st.markdown("""
    El Método de Cauchy es el algoritmo de optimización basado en gradiente más fundamental. Su estrategia es simple e intuitiva: en cada paso, moverse en la dirección en la que la función decrece más rápidamente, que es la dirección opuesta al gradiente (`-∇f(x)`).
    
    1.  Calcular el gradiente `∇f(x)` en el punto actual `xk`.
    2.  Realizar una **búsqueda de línea** (en este caso, con el método de la Sección Dorada) para encontrar la distancia óptima (`α`) a moverse en la dirección `-∇f(x)`.
    3.  Actualizar la posición: `x_{k+1} = xk - α * ∇f(x)`.
    4.  Repetir hasta que la norma del gradiente sea muy pequeña o no haya un cambio significativo.
    
    Este método garantiza la convergencia, pero a menudo lo hace de forma lenta y con un característico patrón en **zigzag**, ya que cada paso es ortogonal al anterior.
    """)

    with st.expander("Configuración del Algoritmo y la Función", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            funcion_seleccionada = st.selectbox("🎯 Selecciona la función:", list(FUNCIONES.keys()), key="cauchy_func")
            info_funcion = FUNCIONES[funcion_seleccionada]
            
            st.markdown("**Parámetros de Parada (Tolerancias):**")
            epsilon1 = st.number_input("ε₁ (Tolerancia de la norma del gradiente)", 0.0, 1.0, 0.001, format="%.5f")
            epsilon2 = st.number_input("ε₂ (Tolerancia de paso relativo / línea)", 0.0, 1.0, 0.001, format="%.5f")

        with col2:
            st.latex(info_funcion['latex'])
            st.markdown("**Parámetros de Ejecución:**")
            max_iter = st.slider("M (Máximo de iteraciones)", 10, 500, 100, 10)
            
            st.markdown("**📍 Punto Inicial (x₀):**")
            bounds_list = info_funcion["intervalos"]
            
            # Crear campos de entrada para cada dimensión de x0
            x0_vals = []
            if bounds_list: # Asegurarse de que bounds_list no esté vacío
                num_dims = len(bounds_list)
                x0_cols = st.columns(num_dims)
                for i, c in enumerate(x0_cols):
                    # Usar el punto medio del intervalo como valor inicial por defecto, si existe.
                    default_val = (bounds_list[i][0] + bounds_list[i][1]) / 2 if bounds_list[i] else 0.0
                    x0_vals.append(c.number_input(f'x0_{i+1}', value=float(default_val), key=f'cauchy_x0_{i}'))
            else:
                st.warning("Define un dominio para la función seleccionada para ingresar el punto inicial.")
                # Si no hay dominio, establecer un x0 predeterminado para evitar errores
                x0_vals = [0.0, 0.0] # Valor por defecto si no hay dominio
                
            x0 = np.array(x0_vals)


    if st.button("🚀 Ejecutar Método de Cauchy", key="run_cauchy"):
        func_objetivo = lambda x: evaluar_funcion(x, funcion_seleccionada)
        
        # Validar que el punto inicial tenga la dimensión correcta
        if len(x0) != len(info_funcion["intervalos"]):
            st.error(f"El punto inicial debe tener {len(info_funcion['intervalos'])} dimensiones para esta función.")
            return

        with st.spinner("Descendiendo por el gradiente..."):
            xk, path_history, table_data = cauchy_algorithm(
                func_objetivo, x0, epsilon1, epsilon2, max_iter
            )
        
        st.success("✅ Optimización completada!")
        
        col1, col2 = st.columns(2)
        # Formato dinámico para la solución encontrada
        if len(xk) == 1:
            col1.metric("🎯 Solución Encontrada", f"[{xk[0]:.4f}]")
        else:
            col1.metric("🎯 Solución Encontrada", f"[{', '.join([f'{val:.4f}' for val in xk])}]")
        col2.metric("📊 Valor Óptimo Encontrado", f"{func_objetivo(xk):.6f}")

        st.markdown("### 📈 Visualización de Resultados")
        
        # Obtener y mostrar todas las figuras
        all_figures = plot_cauchy_results(func_objetivo, info_funcion["intervalos"], path_history, table_data)
        
        for title, fig in all_figures:
            st.subheader(title)
            st.pyplot(fig)
            plt.close(fig) # Cerrar la figura para liberar memoria

        st.markdown("### 📋 Tabla de Iteraciones")
        # Mostrar solo un subconjunto de iteraciones si la tabla es muy larga
        if len(table_data) > 20:
            step = max(1, len(table_data) // 20) # Mostrar ~20 filas
            st.dataframe(pd.DataFrame(table_data[::step]), use_container_width=True)
            st.info(f"Mostrando cada {step}ª iteración de {len(table_data)}.")
        else:
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    # Nuevo apartado con información general sobre el Método de Cauchy
    st.markdown("---") # Separador visual
    st.markdown("### ℹ️ Información General sobre el Método de Cauchy")
    st.markdown("""
    El Método de Cauchy, también conocido como el método del Descenso más Pronunciado o Gradiente Descendente, es uno de los algoritmos fundamentales en la optimización numérica. Se basa en la siguiente lógica:

    * **Principio:** En cada iteración, el algoritmo se mueve en la dirección en la que la función objetivo decrece más rápidamente. Esta dirección es la opuesta al vector gradiente de la función en el punto actual (`-∇f(x)`).
    * **Búsqueda de Línea:** Para determinar cuánto avanzar en esta dirección de máximo descenso, se realiza una "búsqueda de línea" o "line search". Esto implica encontrar un tamaño de paso (`α`) que minimice la función a lo largo de la línea definida por el punto actual y la dirección de descenso. El método de la Sección Dorada es una técnica común para esta búsqueda de línea.
    * **Actualización:** La nueva posición se calcula como `x_{k+1} = xk - α * ∇f(x_k)`.
    * **Convergencia:** Este método está garantizado para converger a un mínimo local para funciones convexas. Sin embargo, su convergencia puede ser lenta, especialmente en funciones con valles estrechos, lo que a menudo se manifiesta como un patrón de "zigzag" en la trayectoria de búsqueda. Esto ocurre porque cada paso es ortogonal al anterior.
    * **Ventajas:** Es simple de entender e implementar, y requiere solo el cálculo de las primeras derivadas (el gradiente).
    * **Desventajas:** Puede ser ineficiente para problemas con muchas variables o funciones mal escaladas, debido a su lenta convergencia y el comportamiento de zigzag. Otros métodos de gradiente, como el Gradiente Conjugado o Newton, a menudo ofrecen una convergencia más rápida.
    """)