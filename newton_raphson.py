import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import io
import tempfile

def metodo_newton_raphson_estricto(f_prime, f_double_prime, x0, epsilon, max_iter=100):
    """
    Implementación estricta del algoritmo de Newton-Raphson para optimización,
    basado en la imagen proporcionada.
    El método busca un punto donde la primera derivada es cero, que puede
    corresponder a un mínimo o un máximo local.
    
    Parámetros:
    - f_prime: La función de la primera derivada de la función a optimizar.
    - f_double_prime: La función de la segunda derivada.
    - x0: El valor inicial de x, correspondiente a x^(1).
    - epsilon: Un número pequeño para el criterio de terminación (tolerancia).
    - max_iter: Número máximo de iteraciones para evitar bucles infinitos.
    
    Retorna:
    - Diccionario con el resultado, historial de iteraciones y información adicional.
    """
    # Step 1: Choose initial guess x^(1) and a small number ε.
    # Set k = 1. Compute f'(x^(1)).
    k = 1
    x_k = x0  # x_k representa x^(k)
    
    # Lista para almacenar el historial de iteraciones
    historial = []
    
    try:
        f_prime_at_xk = f_prime(x_k)  # Calculamos f'(x^(1)) antes de empezar el bucle
    except:
        return {
            'exito': False,
            'mensaje': 'Error al evaluar la primera derivada en el punto inicial',
            'x_optimo': None,
            'historial': []
        }
    
    # Agregar la primera iteración al historial
    historial.append({
        'k': k,
        'x_k': x_k,
        'f_prime_x_k': f_prime_at_xk,
        'f_double_prime_x_k': None,
        'x_k_plus_1': None
    })
    
    # El bucle representa el ciclo desde "go to Step 2"
    while k <= max_iter:
        try:
            # Step 2: Compute f''(x^(k)).
            f_double_prime_at_xk = f_double_prime(x_k)
            
            # Salvaguarda para evitar división por cero
            if abs(f_double_prime_at_xk) < 1e-15:
                return {
                    'exito': False,
                    'mensaje': f'Error: La segunda derivada es prácticamente cero en la iteración {k}. El método no puede continuar.',
                    'x_optimo': None,
                    'historial': historial
                }
            
            # Step 3: Calculate x^(k+1) = x^(k) - f'(x^(k))/f''(x^(k)).
            x_k_plus_1 = x_k - f_prime_at_xk / f_double_prime_at_xk
            
            # Step 3 (continuación): Compute f'(x^(k+1)).
            f_prime_at_xk_plus_1 = f_prime(x_k_plus_1)
            
            # Actualizar el historial con la información completa de esta iteración
            historial[-1]['f_double_prime_x_k'] = f_double_prime_at_xk
            historial[-1]['x_k_plus_1'] = x_k_plus_1
            
            # Step 4: If |f'(x^(k+1))| < ε, Terminate;
            if abs(f_prime_at_xk_plus_1) < epsilon:
                return {
                    'exito': True,
                    'mensaje': f'Convergencia alcanzada en {k} iteraciones',
                    'x_optimo': x_k_plus_1,
                    'f_prime_optimo': f_prime_at_xk_plus_1,
                    'historial': historial,
                    'iteraciones': k
                }
            
            # Step 4 (continuación): Else set k = k + 1 and go to Step 2.
            k = k + 1
            
            # Preparamos los valores para la siguiente iteración
            x_k = x_k_plus_1
            f_prime_at_xk = f_prime_at_xk_plus_1
            
            # Agregar nueva iteración al historial
            historial.append({
                'k': k,
                'x_k': x_k,
                'f_prime_x_k': f_prime_at_xk,
                'f_double_prime_x_k': None,
                'x_k_plus_1': None
            })
            
        except Exception as e:
            return {
                'exito': False,
                'mensaje': f'Error en la iteración {k}: {str(e)}',
                'x_optimo': None,
                'historial': historial
            }
    
    # Si se alcanza el máximo de iteraciones
    return {
        'exito': False,
        'mensaje': f'Se alcanzó el máximo de iteraciones ({max_iter}) sin convergencia',
        'x_optimo': x_k,
        'historial': historial
    }

def obtener_derivadas_funcion(funcion_nombre):
    """
    Retorna las funciones de primera y segunda derivada para las funciones predefinidas.
    """
    derivadas = {
        "Función 1": {
            # f(x) = x^2 + 54/x
            "f_prime": lambda x: 2*x - 54/(x**2),
            "f_double_prime": lambda x: 2 + 108/(x**3),
            "descripcion": "f'(x) = 2x - 54/x², f''(x) = 2 + 108/x³"
        },
        "Función 2": {
            # f(x) = x^3 + 2x - 3
            "f_prime": lambda x: 3*x**2 + 2,
            "f_double_prime": lambda x: 6*x,
            "descripcion": "f'(x) = 3x² + 2, f''(x) = 6x"
        },
        "Función 3": {
            # f(x) = x^4 + x^2 - 33
            "f_prime": lambda x: 4*x**3 + 2*x,
            "f_double_prime": lambda x: 12*x**2 + 2,
            "descripcion": "f'(x) = 4x³ + 2x, f''(x) = 12x² + 2"
        },
        "Función 4": {
            # f(x) = 3x^4 - 8x^3 - 6x^2 + 12x
            "f_prime": lambda x: 12*x**3 - 24*x**2 - 12*x + 12,
            "f_double_prime": lambda x: 36*x**2 - 48*x - 12,
            "descripcion": "f'(x) = 12x³ - 24x² - 12x + 12, f''(x) = 36x² - 48x - 12"
        },
        "Función Lata": {
            # f(x) = 2πx² + 500/x
            "f_prime": lambda x: 4*3.14159*x - 500/(x**2),
            "f_double_prime": lambda x: 4*3.14159 + 1000/(x**3),
            "descripcion": "f'(x) = 4πx - 500/x², f''(x) = 4π + 1000/x³"
        },
        "Función Caja": {
            # f(x) = 4x³ - 60x² + 200x
            "f_prime": lambda x: 12*x**2 - 120*x + 200,
            "f_double_prime": lambda x: 24*x - 120,
            "descripcion": "f'(x) = 12x² - 120x + 200, f''(x) = 24x - 120"
        }
    }
    
    return derivadas.get(funcion_nombre, None)

def plot_newton_raphson_animation(funcion, f_prime, intervalos, historial, fps=1):
    """
    Crea una animación mostrando el proceso del método de Newton-Raphson.
    """
    try:
        # Depuración
        st.write(f"**Depuración** - Tamaño del historial: {len(historial)}")
        st.write(f"**Depuración** - Puntos del historial: {[h['x_k'] for h in historial]}")
        st.write(f"**Depuración** - Intervalos: {intervalos}")

        if len(historial) < 2:
            st.warning("El historial tiene menos de 2 puntos. La animación requiere al menos 2 iteraciones.")
            return None

        # Determinar el rango de la gráfica
        x_opt = historial[-1]['x_k']
        x_min, x_max = intervalos[0][0], intervalos[0][1]
        rango = max(abs(x_opt - x_min), abs(x_max - x_opt))
        x_plot = np.linspace(
            max(x_min, x_opt - rango * 0.5),
            min(x_max, x_opt + rango * 0.5),
            100
        )

        # Evaluar la función
        y_plot = []
        for x in x_plot:
            try:
                y_plot.append(funcion(x))
            except:
                y_plot.append(np.nan)

        if np.all(np.isnan(y_plot)):
            st.error("Error: Todos los valores de la función son NaN o infinitos.")
            return None

        # Crear la figura
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Animación del Método de Newton-Raphson')
        ax.grid(True, alpha=0.3)

        # Inicializar elementos
        initial_point, = ax.plot([], [], 'o', color='lime', markersize=8, label='Punto Inicial')
        current_point, = ax.plot([], [], 'o', color='red', markersize=8, label='Punto Actual')
        final_point, = ax.plot([], [], '*', color='yellow', markersize=12, label='Óptimo')
        tangent_line, = ax.plot([], [], 'orange', linestyle='--', linewidth=1.5, label='Tangente')
        iter_text = ax.text(
            x_plot[0] + 0.05*(x_plot[-1] - x_plot[0]),
            min(y_plot) + 0.05*(max(y_plot) - min(y_plot)),
            '',
            color='white',
            ha='left',
            va='bottom',
            fontsize=10,
            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
        )
        ax.legend()

        def init():
            initial_point.set_data([], [])
            current_point.set_data([], [])
            final_point.set_data([], [])
            tangent_line.set_data([], [])
            iter_text.set_text('')
            return initial_point, current_point, final_point, tangent_line, iter_text

        def update(frame):
            x_k = historial[frame]['x_k']
            f_xk = funcion(x_k) if not np.isnan(funcion(x_k)) else 0
            initial_point.set_data([historial[0]['x_k']], [funcion(historial[0]['x_k'])])
            
            if frame == len(historial) - 1:
                current_point.set_data([], [])
                final_point.set_data([x_k], [f_xk])
            else:
                current_point.set_data([x_k], [f_xk])
                final_point.set_data([], [])
            
            # Calcular la tangente
            try:
                m = f_prime(x_k)
                b = f_xk - m * x_k
                x_tangent = np.array([x_k - rango * 0.1, x_k + rango * 0.1])
                y_tangent = m * x_tangent + b
                tangent_line.set_data(x_tangent, y_tangent)
            except:
                tangent_line.set_data([], [])
            
            iter_text.set_text(f'Iteración {frame}\nx^{frame} = {x_k:.4f}')
            return initial_point, current_point, final_point, tangent_line, iter_text

        # Crear la animación
        st.write("**Depuración** - Creando animación...")
        ani = animation.FuncAnimation(fig, update, frames=len(historial), init_func=init, interval=1000//fps, blit=True)

        # Guardar la animación
        os.makedirs("./animations", exist_ok=True)
        temp_file_path = f"./animations/nr_animation_{np.random.randint(10000)}.gif"
        
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

def show_newton_raphson(FUNCIONES, evaluar_funcion):
    st.markdown("""
    ## 🧮 Método de Newton-Raphson para Optimización
    
    El método de Newton-Raphson es una técnica iterativa que utiliza la primera y segunda derivada 
    de una función para encontrar puntos donde la primera derivada es cero (puntos críticos).
    
    **Fórmula iterativa:** x^(k+1) = x^k - f'(x^k) / f''(x^k)
    
    **Criterio de parada:** |f'(x^(k+1))| < ε
    """)
    
    # Inicializar estado de sesión
    if 'nr_results' not in st.session_state:
        st.session_state.nr_results = None
        st.session_state.funcion_seleccionada = None
        st.session_state.fps = 1

    # Formulario para entrada de parámetros
    with st.form(key="nr_form"):
        st.subheader("⚙️ Configuración")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            funcion_seleccionada = st.selectbox(
                "Selecciona la función a optimizar:",
                list(FUNCIONES.keys()),
                key="nr_funcion"
            )
            st.latex(FUNCIONES[funcion_seleccionada]["latex"])
            st.write(f"**Dominio:** {FUNCIONES[funcion_seleccionada]['dominio']}")
            
            derivadas_info = obtener_derivadas_funcion(funcion_seleccionada)
            if derivadas_info:
                st.write("**Derivadas:**")
                st.write(derivadas_info["descripcion"])
            else:
                st.error("No se han definido las derivadas para esta función.")
                return
        
        with col2:
            x0 = st.number_input(
                "Valor inicial (x₀):",
                value=1.0,
                step=0.1,
                format="%.6f",
                key="nr_x0"
            )
            epsilon = st.number_input(
                "Tolerancia (ε):",
                value=1e-6,
                min_value=1e-12,
                max_value=1e-1,
                format="%.2e",
                key="nr_epsilon"
            )
            max_iter = st.number_input(
                "Máximo de iteraciones:",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                key="nr_max_iter"
            )
            fps = st.slider(
                "Velocidad de la animación (FPS):",
                min_value=1,
                max_value=5,
                value=1,
                key="nr_fps"
            )
        
        submit_button = st.form_submit_button("🚀 Ejecutar Newton-Raphson")

    if submit_button:
        # Validar dominio
        intervalos = FUNCIONES[funcion_seleccionada]["intervalos"]
        dominio_valido = False
        for inicio, fin, _ in intervalos:
            if inicio < x0 <= fin:
                dominio_valido = True
                break
        
        if not dominio_valido:
            st.error(f"El valor inicial x₀ = {x0} está fuera del dominio de la función.")
            return
        
        # Ejecutar el método
        with st.spinner("Calculando iteraciones..."):
            resultado = metodo_newton_raphson_estricto(
                derivadas_info["f_prime"],
                derivadas_info["f_double_prime"],
                x0,
                epsilon,
                max_iter
            )
        
        st.session_state.nr_results = resultado
        st.session_state.funcion_seleccionada = funcion_seleccionada
        st.session_state.fps = fps

    # Mostrar resultados si existen
    if st.session_state.nr_results:
        resultado = st.session_state.nr_results
        funcion_seleccionada = st.session_state.funcion_seleccionada
        fps = st.session_state.fps
        derivadas_info = obtener_derivadas_funcion(funcion_seleccionada)
        
        # Animación
        st.subheader("📽️ Animación del Proceso")
        with st.container():
            gif_output = plot_newton_raphson_animation(
                lambda x: evaluar_funcion(x, funcion_seleccionada),
                derivadas_info["f_prime"],
                FUNCIONES[funcion_seleccionada]["intervalos"],
                resultado['historial'],
                fps
            )
            if gif_output is not None:
                st.image(gif_output, caption="Progreso del algoritmo por iteración", use_container_width=True, output_format='GIF')
                st.success("Animación generada correctamente.")
                st.download_button(
                    label="📥 Descargar Animación",
                    data=gif_output,
                    file_name="nr_animation.gif",
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
                    gif_output = plot_newton_raphson_animation(
                        lambda x: evaluar_funcion(x, funcion_seleccionada),
                        derivadas_info["f_prime"],
                        FUNCIONES[funcion_seleccionada]["intervalos"],
                        resultado['historial'],
                        new_fps
                    )
                    if gif_output is not None:
                        st.image(gif_output, caption="Progreso del algoritmo por iteración", use_container_width=True, output_format='GIF')
                        st.success("Animación regenerada correctamente.")
                        st.download_button(
                            label="📥 Descargar Animación",
                            data=gif_output,
                            file_name="nr_animation.gif",
                            mime="image/gif"
                        )
                    else:
                        st.error("No se pudo regenerar la animación. Revisa los mensajes de depuración.")

        # Resultados numéricos
        st.subheader("📊 Resultados")
        if resultado['exito']:
            st.success(f"✅ {resultado['mensaje']}")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("🎯 Punto óptimo (x*)", f"{resultado['x_optimo']:.8f}")
            with col_res2:
                st.metric("📊 f'(x*)", f"{resultado['f_prime_optimo']:.2e}")
            
            try:
                f_optimo = evaluar_funcion(resultado['x_optimo'], funcion_seleccionada)
                st.metric("📈 Valor de la función f(x*)", f"{f_optimo:.8f}")
            except:
                st.warning("No se pudo evaluar f(x*)")
        else:
            st.error(f"❌ {resultado['mensaje']}")
            if resultado['x_optimo'] is not None:
                st.info(f"Último valor calculado: x = {resultado['x_optimo']:.8f}")
        
        # Tabla de iteraciones
        if resultado['historial']:
            st.subheader("📋 Historial de Iteraciones")
            tabla_data = []
            for iteracion in resultado['historial']:
                fila = {
                    'k': iteracion['k'],
                    'x^k': f"{iteracion['x_k']:.8f}",
                    "f'(x^k)": f"{iteracion['f_prime_x_k']:.6e}",
                    "|f'(x^k)|": f"{abs(iteracion['f_prime_x_k']):.6e}"
                }
                if iteracion['f_double_prime_x_k'] is not None:
                    fila["f''(x^k)"] = f"{iteracion['f_double_prime_x_k']:.6f}"
                else:
                    fila["f''(x^k)"] = "-"
                if iteracion['x_k_plus_1'] is not None:
                    fila['x^(k+1)'] = f"{iteracion['x_k_plus_1']:.8f}"
                else:
                    fila['x^(k+1)'] = "-"
                tabla_data.append(fila)
            
            df = pd.DataFrame(tabla_data)
            st.dataframe(df, use_container_width=True)
        
        # Gráfica estática
        if resultado['exito'] and resultado['x_optimo'] is not None:
            st.subheader("📈 Visualización Gráfica")
            try:
                x_opt = resultado['x_optimo']
                intervalo = FUNCIONES[funcion_seleccionada]["intervalos"][0]
                x_min, x_max = intervalo[0], intervalo[1]
                rango = max(abs(x_opt - x_min), abs(x_max - x_opt))
                x_plot = np.linspace(
                    max(x_min, x_opt - rango * 0.5),
                    min(x_max, x_opt + rango * 0.5),
                    100
                )
                y_plot = []
                for x in x_plot:
                    try:
                        y_plot.append(evaluar_funcion(x, funcion_seleccionada))
                    except:
                        y_plot.append(np.nan)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')
                f_opt = evaluar_funcion(x_opt, funcion_seleccionada)
                ax.plot(x_opt, f_opt, 'ro', markersize=10, label=f'Óptimo: ({x_opt:.4f}, {f_opt:.4f})')
                f_x0 = evaluar_funcion(x0, funcion_seleccionada)
                ax.plot(x0, f_x0, 'go', markersize=8, label=f'Inicial: ({x0:.4f}, {f_x0:.4f})')
                ax.set_xlabel('x')
                ax.set_ylabel('f(x)')
                ax.set_title(f'Método de Newton-Raphson - {funcion_seleccionada}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Error al crear la gráfica: {str(e)}")
    
    # Información adicional
    with st.expander("ℹ️ Información sobre el Método de Newton-Raphson"):
        st.markdown("""
        ### Características del Método:
        
        **Ventajas:**
        - Convergencia cuadrática (muy rápida) cerca del óptimo
        - Pocas iteraciones necesarias cuando converge
        - Utiliza información de segunda derivada
        
        **Desventajas:**
        - Requiere calcular la segunda derivada
        - Puede no converger si f''(x) = 0
        - Sensible al punto inicial
        - Solo encuentra óptimos locales
        
        **Algoritmo:**
        1. Elegir punto inicial x⁽¹⁾ y tolerancia ε
        2. Para k = 1, 2, 3, ...
        3. Calcular f''(x⁽ᵏ⁾)
        4. Si f''(x⁽ᵏ⁾) = 0, terminar con error
        5. Calcular x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ - f'(x⁽ᵏ⁾)/f''(x⁽ᵏ⁾)
        6. Si |f'(x⁽ᵏ⁺¹⁾)| < ε, terminar con éxito
        7. Continuar con k = k + 1
        
        **Interpretación Geométrica:**
        El método utiliza la aproximación cuadrática de Taylor de la función 
        en cada punto para encontrar donde la derivada es cero.
        """)