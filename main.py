import streamlit as st

st.set_page_config(
    page_title="Proyecto de Optimización",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np
from busqueda_exhaustiva import show_busqueda_exhaustiva
from fase_acotamiento import show_fase_acotamiento
from region_elimination import show_region_elimination
from intervalos_mitad import show_intervalos_mitad
from fibonacci_search import show_fibonacci_search
from golden_section_search import show_golden_section_search
from newton_raphson import show_newton_raphson
from metodo_biseccion import show_biseccion_search
from metodo_secante import show_metodo_secante
from nelder_mead import show_nelder_mead
from hooke_jeeves import show_hooke_jeeves
from random_walk import show_random_walk
from hill_climbing import show_hill_climbing
from simulated_annealing import show_simulated_annealing
from cauchy_method import show_cauchy_method
from newton_method import show_newton_method
from conjugate_gradient import show_conjugate_gradient
from unidirectional_search import show_busqueda_unidireccional

FUNCIONES_UNIVARIADAS = {
    "Función 1": {
        "intervalos": [(0.001, 10, lambda x: x**2 + 54/x)], # Ajustado el inicio a 0.001 para evitar división por cero
        "latex": r'''f(x) = x^2 + \frac{54}{x}''',
        "latex_derivada": r'''f'(x) = 2x - \frac{54}{x^2}''',
        "dominio": "(0, 10]"
    },
    "Función 2": {
        "intervalos": [(0, 5, lambda x: x**3 + 2*x - 3)],
        "latex": r'''f(x) = x^3 + 2x - 3''',
        "latex_derivada": r'''f'(x) = 3x^2 + 2''',
        "dominio": "(0, 5]"
    },
    "Función 3": {
        "intervalos": [(-2.5, 2.5, lambda x: x**4 + x**2 - 33)],
        "latex": r'''f(x) = x^4 + x^2 - 33''',
        "latex_derivada": r'''f'(x) = 4x^3 + 2x''',
        "dominio": "[-2.5, 2.5]"
    },
    "Función 4": {
        "intervalos": [(-1.5, 3, lambda x: 3*x**4 - 8*x**3 - 6*x**2 + 12*x)],
        "latex": r'''f(x) = 3x^4 - 8x^3 - 6x^2 + 12x''',
        "latex_derivada": r'''f'(x) = 12x^3 - 24x^2 - 12x + 12''',
        "dominio": "[-1.5, 3]"
    },
    "Función Lata": {
        "intervalos": [(0.001, 10, lambda x: 2*np.pi*x*x + (500 / x))], # Ajustado el inicio a 0.001
        "latex": r'''f(x) = 2\pi r^2 + \frac{500}{r}''',
        "latex_derivada": r'''f'(x) = 4\pi r - \frac{500}{r^2}''',
        "dominio": "(0, ∞)"
    },
    "Función Caja": {
        "intervalos": [(0.1, 10, lambda x: 4*x**3 - 60*x**2 + 200*x)], # Se asume una función de volumen de caja correcta
        "latex": r'''f(x) = 4x^3 - 60x^2 + 200x''', # Formato para la caja, asumimos que x es la altura
        "latex_derivada": r'''f'(x) = 12x^2 - 120x + 200''',
        "dominio": "(0, ∞)"
    }
}

# Modificado para manejar errores de división por cero o fuera de dominio
def evaluar_funcion_univariada(x, funcion_nombre):
    intervalos_info = FUNCIONES_UNIVARIADAS[funcion_nombre]["intervalos"]

    for inicio, fin, func in intervalos_info:
        # Manejo específico para funciones con división por cero en x=0
        if (funcion_nombre == "Función 1" or funcion_nombre == "Función Lata") and x == 0:
            return np.inf # Un valor "malo" para la minimización

        if inicio <= x <= fin:
            try:
                return func(x)
            except (ValueError, ZeroDivisionError):
                return np.inf # Retorna infinito si hay un error matemático
    
    # En caso de que x esté fuera de los intervalos definidos (lo cual no debería pasar si los intervalos cubren el dominio)
    # o si la lógica anterior no capturó el error
    try:
        # Intentar evaluar con la primera función si no se encontró en el bucle
        return intervalos_info[0][2](x)
    except (ValueError, ZeroDivisionError):
        return np.inf # Retorna infinito si hay un error matemático inesperado


def evaluar_derivada(x, funcion_nombre):
    derivadas = {
        "Función 1": lambda x: 2*x - 54/(x**2),
        "Función 2": lambda x: 3*x**2 + 2,
        "Función 3": lambda x: 4*x**3 + 2*x,
        "Función 4": lambda x: 12*x**3 - 24*x**2 - 12*x + 12, # Corregida la derivada de Función 4
        "Función Lata": lambda x: 4*np.pi*x - 500/(x**2),
        "Función Caja": lambda x: 12*x**2 - 120*x + 200 # Corregida la derivada de Función Caja
    }
    if funcion_nombre in derivadas:
        try:
            # Manejo específico para derivadas con división por cero en x=0
            if (funcion_nombre == "Función 1" or funcion_nombre == "Función Lata") and x == 0:
                raise ZeroDivisionError("Derivada indefinida en x=0")
            return derivadas[funcion_nombre](x)
        except (ValueError, ZeroDivisionError) as e:
            raise ValueError(f"Error al evaluar la derivada en x = {x}: {e}")
    else:
        raise ValueError(f"Derivada no definida para la función: {funcion_nombre}")

def evaluar_funcion_multivariada(x, funcion_nombre):
    if funcion_nombre in FUNCIONES_MULTIVARIADAS:
        try: return FUNCIONES_MULTIVARIADAS[funcion_nombre]["funcion"](x)
        except Exception as e:
            st.error(f"Error al evaluar la función multivariada '{funcion_nombre}' en el punto {x}: {e}")
            return None
    else: raise ValueError(f"Función multivariada no encontrada: {funcion_nombre}")

FUNCIONES_MULTIVARIADAS = {
    "Rastrigin": {
        "funcion": lambda x: 10 * 2 + (x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1])),
        "gradiente": lambda x: np.array([2*x[0] + 20*np.pi*np.sin(2*np.pi*x[0]), 2*x[1] + 20*np.pi*np.sin(2*np.pi*x[1])]),
        "intervalos": [(-5.12, 5.12), (-5.12, 5.12)],
        "latex": r"f(x, y) = 20 + (x^2 - 10 \cos(2\pi x)) + (y^2 - 10 \cos(2\pi y))",
        "dominio": "[-5.12, 5.12] x [-5.12, 5.12]",
        "minimo": "f(0, 0) = 0",
        "minimo_coords": np.array([0.0, 0.0])
    },
    "Ackley": {
        "funcion": lambda x: -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20,
        "gradiente": lambda x: np.array([
            (2*x[0]*np.exp(-0.2*np.sqrt(0.5*(x[0]**2+x[1]**2))))/(np.sqrt(0.5*(x[0]**2+x[1]**2))+1e-9) + np.pi*np.sin(2*np.pi*x[0])*np.exp(0.5*(np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1]))),
            (2*x[1]*np.exp(-0.2*np.sqrt(0.5*(x[0]**2+x[1]**2))))/(np.sqrt(0.5*(x[0]**2+x[1]**2))+1e-9) + np.pi*np.sin(2*np.pi*x[1])*np.exp(0.5*(np.cos(2*np.pi*x[1])+np.cos(2*np.pi*x[0]))) # Corregida la derivada para evitar repetición
        ]),
        "intervalos": [(-5, 5), (-5, 5)],
        "latex": r"f(x,y) = -20e^{-0.2\sqrt{0.5(x^2+y^2)}} - e^{0.5(\cos(2\pi x)+\cos(2\pi y))} + e + 20",
        "dominio": "[-5, 5] x [-5, 5]",
        "minimo": "f(0, 0) = 0",
        "minimo_coords": np.array([0.0, 0.0])
    },
    "Sphere": {
        "funcion": lambda x: x[0]**2 + x[1]**2,
        "gradiente": lambda x: np.array([2*x[0], 2*x[1]]),
        "intervalos": [(-10, 10), (-10, 10)],
        "latex": r"f(x, y) = x^2 + y^2",
        "dominio": "[-10, 10] x [-10, 10]",
        "minimo": "f(0, 0) = 0",
        "minimo_coords": np.array([0.0, 0.0])
    },
    "Rosenbrock": {
        "funcion": lambda x: 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2,
        "gradiente": lambda x: np.array([-400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0]), 200*(x[1]-x[0]**2)]),
        "intervalos": [(-2, 2), (-1, 3)],
        "latex": r"f(x, y) = 100(y - x^2)^2 + (1 - x)^2",
        "dominio": "[-2, 2] x [-1, 3]",
        "minimo": "f(1, 1) = 0",
        "minimo_coords": np.array([1.0, 1.0])
    },
    "Beale": {
        "funcion": lambda x: (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2,
        "gradiente": lambda x: np.array([2*(1.5 - x[0] + x[0]*x[1])*(-1 + x[1]) + 2*(2.25 - x[0] + x[0]*x[1]**2)*(-1 + x[1]**2) + 2*(2.625 - x[0] + x[0]*x[1]**3)*(-1 + x[1]**3), 2*(1.5 - x[0] + x[0]*x[1])*x[0] + 2*(2.25 - x[0] + x[0]*x[1]**2)*(2*x[0]*x[1]) + 2*(2.625 - x[0] + x[0]*x[1]**3)*(3*x[0]*x[1]**2)]),
        "intervalos": [(-4.5, 4.5), (-4.5, 4.5)],
        "latex": r"f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2",
        "dominio": "[-4.5, 4.5] x [-4.5, 4.5]",
        "minimo": "f(3, 0.5) = 0",
        "minimo_coords": np.array([3.0, 0.5])
    },
    "Booth": {
        "funcion": lambda x: (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2,
        "gradiente": lambda x: np.array([2*(x[0] + 2*x[1] - 7) + 4*(2*x[0] + x[1] - 5), 4*(x[0] + 2*x[1] - 7) + 2*(2*x[0] + x[1] - 5)]),
        "intervalos": [(-10, 10), (-10, 10)],
        "latex": r"f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2",
        "dominio": "[-10, 10] x [-10, 10]",
        "minimo": "f(1, 3) = 0",
        "minimo_coords": np.array([1.0, 3.0])
    },
    "Himmelblau": {
        "funcion": lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2,
        "gradiente": lambda x: np.array([4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7), 2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)]),
        "intervalos": [(-5, 5), (-5, 5)],
        "latex": r"f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2",
        "dominio": "[-5, 5] x [-5, 5]",
        "minimo": "Varios mínimos, ej: f(3, 2) = 0",
        "minimo_coords": np.array([3.0, 2.0])
    },
    "McCormick": {
        "funcion": lambda x: np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1,
        "gradiente": lambda x: np.array([np.cos(x[0] + x[1]) + 2*(x[0] - x[1]) - 1.5, np.cos(x[0] + x[1]) - 2*(x[0] - x[1]) + 2.5]),
        "intervalos": [(-1.5, 4), (-3, 4)],
        "latex": r"f(x, y) = \sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1",
        "dominio": "[-1.5, 4] x [-3, 4]",
        "minimo": "f(-0.547, -1.547) ≈ -1.913",
        "minimo_coords": np.array([-0.547, -1.547])
    }
}


# --- ESTILOS CON CSS ---
st.markdown("""
<style>
/* --- INICIO DE NUEVOS ESTILOS Y MEJORAS --- */

/* --- Estilos Globales --- */
body {
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', sans-serif;
    color: #333;
    background-color: #f0f2f6; /* Un gris muy claro para el fondo principal */
}

/* --- Títulos y Texto --- */
h1, h2, h3, h4, h5, h6 {
    color: #1a252f;
}

h2 {
    border-bottom: 2px solid #667eea;
    padding-bottom: 10px;
    margin-top: 40px;
}

h3 {
    color: #4a4a4a;
    margin-top: 30px;
}

/* --- Barra Lateral (Sidebar) --- */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e6e6e6;
}
.sidebar-title {
    font-size: 24px;
    font-weight: bold;
    color: #667eea;
    text-align: center;
    margin-bottom: 20px;
    padding: 10px;
    border-bottom: 2px solid #764ba2;
}
.author-name {
    font-size: 16px;
    color: #7f8c8d;
    text-align: center;
    margin-top: 20px;
    font-style: italic;
}
/* Estilo para los radio buttons del menú principal */
div[role="radiogroup"] > label {
    display: block;
    padding: 10px 15px;
    margin: 5px 0;
    border-radius: 8px;
    border: 1px solid #ddd;
    transition: all 0.2s ease-in-out;
}
div[role="radiogroup"] > label:hover {
    background-color: #f0f2f6;
    border-color: #667eea;
}

/* --- Título Principal --- */
.main-title {
    font-size: 42px;
    font-weight: bold;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 30px;
    padding: 20px;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.05);
}

/* --- Contenedores y Tarjetas --- */
.grid-background {
    font-size: 18px;
    color: #34495e;
    text-align: left; /* Mejor para leer */
    padding: 40px;
    background-color: #ffffff;
    border-radius: 12px;
    border: 1px solid #e6e6e6;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin: 20px 0;
}
/* Estilo para st.metric */
[data-testid="stMetric"] {
    background-color: #ffffff;
    border-left: 5px solid #667eea;
    padding: 15px 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
[data-testid="stMetric"] > div:nth-child(2) {
    color: #667eea !important;
}

/* --- Botones --- */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px;
    margin: 5px 0;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(102,126,234,0.4);
}
.stButton > button:active {
    transform: translateY(0);
}

/* --- Otros Componentes de Streamlit --- */
/* Mensajes de información */
[data-testid="stInfo"], [data-testid="stSuccess"], [data-testid="stWarning"], [data-testid="stError"] {
    border-radius: 8px;
    border-left-width: 5px;
    padding: 15px;
}
[data-testid="stExpander"] {
    border: 1px solid #e6e6e6;
    border-radius: 8px;
    background-color: #ffffff;
}
[data-testid="stExpander"] > details > summary {
    font-size: 1.1em;
    font-weight: bold;
    color: #2c3e50;
}
/* Bloques de código y LaTeX */
div[data-testid="stCodeBlock"], .stLatex {
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

/* --- FIN DE NUEVOS ESTILOS --- */
</style>
""", unsafe_allow_html=True)

# --- BARRA LATERAL ---
with st.sidebar:
    st.markdown('<div class="sidebar-title">Proyecto Final de Optimización</div>', unsafe_allow_html=True)

    st.session_state.main_view = st.radio(
        "Menú Principal",
        ["Página Principal", "Contexto de la Materia", "Algoritmos de Optimización"],
        key="main_nav"
    )

    if st.session_state.main_view == "Algoritmos de Optimización":
        st.markdown("---")
        st.markdown("### Métodos Univariados")

        univariados_options = [
            "Búsqueda Exhaustiva",
            "Método de fase de acotamiento",
            "Método de Eliminación de Regiones",
            "Intervalos por la mitad",
            "Fibonacci Search Method",
            "Golden Section Search Method",
            "Método de Newton-Raphson",
            "Método de Bisección",
            "Método de la secante"
        ]
        for option in univariados_options:
            if st.button(option, key=f"btn_{option}"):
                st.session_state.current_page = option

        st.markdown("### Métodos Multivariados")
        multivariados_options = [
            "Búsqueda Unidireccional",
            "Nelder-Mead Simplex", "Hooke-Jeeves - Movimiento exploratorio", "Random Walk (Caminata Aleatoria)",
            "Ascenso/Descenso de la colina (Hill Climbing)", "Recocido Simulado (Simulated Annealing)",
            "Método de Cauchy", "Método de Newton", "Gradiente Conjugado"
        ]
        for option in multivariados_options:
            if st.button(option, key=f"btn_{option}"):
                st.session_state.current_page = option

    st.markdown('<div class="author-name">😎 Oscar David López Ibarra 🤠</div>', unsafe_allow_html=True)


# --- ÁREA DE CONTENIDO PRINCIPAL ---

if st.session_state.main_view != "Algoritmos de Optimización":
    st.session_state.current_page = "Página Principal"

# Vista de Página Principal
if st.session_state.main_view == "Página Principal":
    st.markdown(f'<h1 class="main-title">Proyecto Final de Optimización</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="grid-background">
        <h2>Bienvenido a la Aplicación Interactiva de Métodos de Optimización</h2>
        <p>Esta plataforma ha sido diseñada para explorar, ejecutar y visualizar una amplia gama de algoritmos de optimización, tanto univariados como multivariados.</p>
        <p>Utiliza el menú lateral para navegar entre las diferentes secciones:</p>
        <ul>
            <li><b>Contexto de la Materia:</b> Aprende los conceptos teóricos fundamentales de la optimización.</li>
            <li><b>Algoritmos de Optimización:</b> Experimenta con cada uno de los métodos implementados.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Vista de Contexto de la Materia
elif st.session_state.main_view == "Contexto de la Materia":
    st.markdown(f'<h1 class="main-title">Contexto de la Materia de Optimización</h1>', unsafe_allow_html=True)
    st.markdown("Un resumen de los conceptos clave presentados en el material de clase.")

    st.header("¿Qué es la Optimización?")
    st.write("""
    Matemáticamente, la optimización consiste en encontrar la **mejor solución posible** para un problema, lo cual se logra cambiando el valor de variables que se pueden controlar. Este proceso a menudo está sujeto a un conjunto de restricciones.

    Cualquier problema que requiera la toma de una decisión puede ser formulado como un problema de optimización. Su atractivo es universal, ya que se aplica en todos los dominios del conocimiento y responde al deseo humano de mejorar las cosas.
    """)

    st.header("Componentes de un Problema de Optimización")
    st.markdown("""
    Todo problema de optimización se compone de los siguientes elementos:
    - **Función Objetivo:** Es la función que se desea minimizar o maximizar. Debe ser un valor escalar que se calcula a partir de las variables de decisión. Por convención, los problemas de maximización se convierten en problemas de minimización de la siguiente forma: `max[f(x)] = -min[-f(x)]`.
    - **Variables de Decisión:** Son las variables que describen el sistema y que se pueden controlar o modificar. Se representan comúnmente como un vector.
    - **Límites (Bounds):** Definen los valores máximos y mínimos que cada variable de decisión puede tomar. Esta información ayuda a acotar el espacio de búsqueda para los algoritmos.
    - **Restricciones:** Son relaciones funcionales que deben satisfacerse. Definen la **región factible**, que es el conjunto de todas las soluciones que cumplen con las restricciones. Existen dos tipos: de igualdad ($h_k(x)=0$) y de desigualdad ($g_j(x) \ge 0$).
    """)

    st.header("Formas de Abordar la Optimización")
    st.write("Existen cuatro enfoques principales para resolver un problema de optimización:")

    st.subheader("1. Métodos Analíticos")
    st.write("Basados en el cálculo diferencial clásico. Se busca el punto donde las derivadas de la función objetivo son cero. No requieren una computadora, pero su aplicación es limitada a problemas con pocas variables y sin alta no-linealidad.")

    st.subheader("2. Métodos Gráficos")
    st.write("Requieren dibujar la función a optimizar. Son muy útiles para visualizar el problema, pero su utilidad se limita a problemas con una o dos variables.")

    st.subheader("3. Métodos Experimentales")
    st.write("Consisten en ajustar los valores de las variables de forma secuencial y evaluar el resultado en cada paso. Pueden acercarse al óptimo, pero pueden fallar si las variables interactúan entre sí y necesitan ser ajustadas simultáneamente.")

    st.subheader("4. Métodos Numéricos")
    st.write("""
    Son el enfoque más importante y general. Utilizan un procedimiento iterativo para generar una serie de soluciones que mejoran progresivamente, partiendo de una estimación inicial. El proceso finaliza cuando se alcanza un criterio de convergencia.

    Estos métodos pueden resolver problemas altamente complejos que no son tratables analíticamente y son fáciles de programar en una computadora, lo que ha llevado a que reemplacen casi por completo a los otros enfoques. La disciplina que los estudia se conoce como **Programación Matemática**.
    """)

    st.header("Conceptos Clave Adicionales")
    st.markdown("""
    - **Óptimo Local vs. Global:** Un punto es un óptimo local si es el mejor en su vecindario inmediato. Un punto es un óptimo global si es el mejor en todo el espacio de búsqueda.
    - **Funciones Unimodales y Multimodales:** Las funciones unimodales tienen un solo mínimo, mientras que las multimodales tienen múltiples mínimos.
    - **Convexidad:** Una función es convexa si el segmento que une dos puntos cualesquiera de su gráfica queda por encima de la misma. Las funciones convexas tienen la valiosa propiedad de que cualquier mínimo local es también un mínimo global.
    """)

# Vista de Algoritmos de Optimización
elif st.session_state.main_view == "Algoritmos de Optimización":
    if 'current_page' not in st.session_state or st.session_state.current_page == "Página Principal":
        st.session_state.current_page = "Algoritmos de Optimización"

    st.markdown(f'<h1 class="main-title">{st.session_state.current_page}</h1>', unsafe_allow_html=True)

    if st.session_state.current_page == "Algoritmos de Optimización":
        st.markdown("""
        <div class="grid-background">
            <h2>Bienvenido a la Sección de Algoritmos</h2>
            <p>Selecciona un método del menú lateral para comenzar a experimentar.</p>
            <p>Puedes elegir entre métodos univariados (una variable) y multivariados (múltiples variables).</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("## Métodos Disponibles")

    # --- Lógica de enrutamiento para cada algoritmo ---
    elif st.session_state.current_page == "Búsqueda Unidireccional":
        show_busqueda_unidireccional(FUNCIONES_MULTIVARIADAS, evaluar_funcion_multivariada)
    elif st.session_state.current_page == "Búsqueda Exhaustiva":
        show_busqueda_exhaustiva(FUNCIONES_UNIVARIADAS, evaluar_funcion_univariada)
    elif st.session_state.current_page == "Método de fase de acotamiento":
        show_fase_acotamiento(FUNCIONES_UNIVARIADAS, evaluar_funcion_univariada)
    elif st.session_state.current_page == "Método de Eliminación de Regiones":
        show_region_elimination(FUNCIONES_UNIVARIADAS, evaluar_funcion_univariada)
    elif st.session_state.current_page == "Intervalos por la mitad":
        show_intervalos_mitad(FUNCIONES_UNIVARIADAS, evaluar_funcion_univariada)
    elif st.session_state.current_page == "Fibonacci Search Method":
        show_fibonacci_search(FUNCIONES_UNIVARIADAS, evaluar_funcion_univariada)
    elif st.session_state.current_page == "Golden Section Search Method":
        show_golden_section_search(FUNCIONES_UNIVARIADAS, evaluar_funcion_univariada)
    elif st.session_state.current_page == "Método de Newton-Raphson":
        show_newton_raphson(FUNCIONES_UNIVARIADAS, evaluar_funcion_univariada)
    elif st.session_state.current_page == "Método de Bisección":
        show_biseccion_search(FUNCIONES_UNIVARIADAS, evaluar_funcion_univariada, evaluar_derivada)
    elif st.session_state.current_page == "Método de la secante":
        show_metodo_secante(FUNCIONES_UNIVARIADAS, evaluar_funcion_univariada)
    elif st.session_state.current_page == "Nelder-Mead Simplex":
        show_nelder_mead(FUNCIONES_MULTIVARIADAS, evaluar_funcion_multivariada)
    elif st.session_state.current_page == "Hooke-Jeeves - Movimiento exploratorio":
        show_hooke_jeeves(FUNCIONES_MULTIVARIADAS, evaluar_funcion_multivariada)
    elif st.session_state.current_page == "Random Walk (Caminata Aleatoria)":
        show_random_walk(FUNCIONES_MULTIVARIADAS, evaluar_funcion_multivariada)
    elif st.session_state.current_page == "Ascenso/Descenso de la colina (Hill Climbing)":
        show_hill_climbing(FUNCIONES_MULTIVARIADAS, evaluar_funcion_multivariada)
    elif st.session_state.current_page == "Recocido Simulado (Simulated Annealing)":
        show_simulated_annealing(FUNCIONES_MULTIVARIADAS, evaluar_funcion_multivariada)
    elif st.session_state.current_page == "Método de Cauchy":
        show_cauchy_method(FUNCIONES_MULTIVARIADAS, evaluar_funcion_multivariada)
    elif st.session_state.current_page == "Método de Newton":
        show_newton_method(FUNCIONES_MULTIVARIADAS, evaluar_funcion_multivariada)
    elif st.session_state.current_page == "Gradiente Conjugado":
        show_conjugate_gradient(FUNCIONES_MULTIVARIADAS)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; margin-top: 20px;'>
    © 2025 - Proyecto Final de Optimización | Desarrollado con ❤️ usando Streamlit para la materia de Optimización
</div>
""", unsafe_allow_html=True)