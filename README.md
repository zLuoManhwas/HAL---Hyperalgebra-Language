# HAL — Hyperalgebra Language

Prototipo modular en Python para describir composiciones categóricas y cálculos sobre hipergrafos. Incluye un parser tipado, reescrituras locales de álgebra de Frobenius, matrices de incidencia NumPy/SciPy, tensores para hipergrafos uniformes y un kernel de agregación dispersa para CPU/CUDA.

> **Estado:** prototipo verificable, no compilador de producción. El normalizador no decide toda equivalencia categórica; el kernel JIT implementado ejecuta una agregación CSR concreta, no compila expresiones HAL arbitrarias. La ejecución CUDA necesita un dispositivo y runtime CUDA compatibles. Consulta [HAL_SPECIFICATION.md](HAL_SPECIFICATION.md) para la especificación y limitaciones detalladas.

## ¿Qué puedes hacer con HAL hoy?

HAL resulta útil cuando una relación conecta a un grupo completo y no solo a pares de elementos. Puedes modelar esa relación como una hiperarista, inspeccionar sus incidencias y calcular proyecciones o agregaciones ponderadas con las herramientas que ya incluye el proyecto.

| Caso práctico | Cómo aprovechar HAL ahora |
|---|---|
| Colaboraciones entre varias personas | Representa cada equipo, artículo o proyecto como una hiperarista que reúne a todos sus participantes; calcula la incidencia o la proyección de co-colaboración. |
| Eventos con muchos asistentes | Codifica cada evento como una hiperarista y analiza qué participantes coinciden en eventos, conservando pesos por importancia o frecuencia. |
| Interacciones multicomponente | Modela una relación que involucra simultáneamente varios nodos sin descomponerla primero en aristas binarias; usa el tensor simétrico si todas las hiperaristas tienen el mismo tamaño. |
| Propagación de valores sobre grupos | Asigna un valor a cada hiperarista y ejecuta la agregación CSR ponderada de hiperaristas a vértices, con la versión NumPy de referencia o el JIT CPU de Numba. |
| Validación de redes de operaciones | Escribe composiciones con `#` y `@`; el comprobador detecta conexiones cuyos tipos intermedios no coinciden antes de ejecutar el análisis. |
| Probar identidades de diagramas | Aplica el normalizador local para reducir identidades, especialidad y patrones implementados de Frobenius antes de comparar o inspeccionar expresiones. |

### La ventaja práctica

HAL ofrece una forma única de describir y analizar interacciones de orden superior: el modelo conserva qué elementos pertenecen juntos a cada relación, mientras que la sintaxis categórica permite expresar conexiones secuenciales y paralelas sin escribir a mano índices tensoriales. El chequeo de tipos ayuda a detectar errores de composición; la incidencia dispersa evita construir tensores densos cuando las hiperaristas son heterogéneas; y el tensor uniforme permite usar una representación directa cuando las interacciones sí comparten tamaño. Para agregación CSR, puedes empezar con una implementación de referencia y activar JIT CPU con Numba sin cambiar el modelo del hipergrafo.

Esto es especialmente atractivo para prototipar análisis de colaboración, co-ocurrencia y propagación sobre grupos, y para experimentar con una semántica algebraica verificable. **No** ofrece todavía entrenamiento de redes neuronales, inferencia automática sobre cualquier programa HAL, compilación general de expresiones a GPU ni garantías de rendimiento de producción; la ejecución CUDA disponible se limita al kernel CSR incluido y requiere hardware/software compatible.

## Funcionalidad comprobada

- Parser con precedencia de operadores, tipos, declaraciones de morfismos, comentarios y errores con ubicación.
- Inferencia de dominio/codominio con unificación de tipos, variables polimórficas frescas y detección de composición incompatible.
- Reglas estructurales locales de identidad, asociatividad/coasociatividad, Frobenius, unidad/counidad, especialidad y simetría.
- Matriz de incidencia densa NumPy o dispersa CSR con SciPy; proyección ponderada `B W Bᵀ`, diagonal de grados y adyacencia sin diagonal.
- Tensor simétrico de adyacencia para hipergrafos uniformes en formato denso acotado o COO disperso.
- Agregación CSR de referencia, compilación JIT CPU con Numba cuando está instalado, generador de fuente Python y kernel CUDA Numba/fuente CUDA.

## Requisitos

- Python 3.10 o posterior.
- NumPy y SciPy para los backends matriciales.
- Numba opcional para JIT CPU y ejecución CUDA.
- GPU, controlador/runtime y entorno compatibles para ejecutar CUDA.

## Instalación

En PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[all]"
```

Instalación solo de dependencias requeridas:

```powershell
python -m pip install -e .
```

Instalar desde el código fuente sin empaquetarlo:

```powershell
python -m pip install -r requirements.txt
# Opcional: Numba para JIT
python -m pip install -r requirements-jit.txt
```

## Inicio rápido

```python
from hal import BaseType, infer_type, parse_hal, tensor

signatures = {
    "f": (BaseType("A"), tensor(BaseType("B"), BaseType("C"))),
    "g": (BaseType("B"), BaseType("D")),
}

program = parse_hal(
    "f # (split(B) @ id(C)) # (g @ id(B) @ id(C))",
    signatures,
)

print(program)
print(infer_type(program))
```

La composición `#` se escribe de izquierda a derecha: `e1 # e2` significa ejecutar `e1` y después `e2`. `@` compone en paralelo y tiene precedencia mayor que `#`. Las composiciones solo son válidas cuando los tipos intermedios coinciden.

Los morfismos deben declararse inline (`f: A -> B`) o proporcionarse en el registro. Los nombres de tipos en minúscula denotan variables de tipo; los nombres en mayúscula son tipos concretos.

## Cálculo con hipergrafos

```python
from hal import Hyperedge, Hypergraph

graph = Hypergraph(
    vertices=("u", "v", "w"),
    edges=(
        Hyperedge("e0", ("u", "v", "w"), weight=1.0),
        Hyperedge("e1", ("v", "w"), weight=2.0),
    ),
)

B = graph.incidence_matrix()               # NumPy densa
projection = graph.clique_projection()     # NumPy densa
print(B)
print(projection.gram)       # B W B.T
print(projection.adjacency)  # B W B.T menos la diagonal

# Hipergrafo uniforme: tensor de adyacencia totalmente simétrico.
uniform = Hypergraph(
    vertices=("u", "v", "w"),
    edges=(Hyperedge("e0", ("u", "v", "w")),),
)
dense_tensor = uniform.adjacency_tensor()
coo_tensor = uniform.adjacency_tensor(format="coo")
```

Para estructuras grandes, usa incidencia CSR:

```python
incidence_csr = graph.incidence_matrix(format="csr")
projection_csr = graph.clique_projection(format="csr")
```

SciPy debe estar instalado para CSR. La materialización densa del tensor tiene un límite por defecto de 10 millones de elementos; el formato COO tiene un límite de expansión de un millón de entradas simétricas. Ambos límites pueden ajustarse pasando `max_elements` y `max_entries`.

## Kernel de agregación CSR

La operación implementada propaga valores de hiperaristas a vértices:

```text
y[v] = Σ_e B[v,e] · weight[e] · value[e]
```

```python
import numpy as np
from hal import Hyperedge, Hypergraph, jit_cpu_aggregate

graph = Hypergraph(
    ("u", "v", "w"),
    (Hyperedge("e0", ("u", "v"), 2.0), Hyperedge("e1", ("v", "w"), 3.0)),
)
csr = graph.incidence_matrix("csr")

kernel = jit_cpu_aggregate()
y = kernel(
    csr.indptr,
    csr.indices,
    csr.data,
    np.array([2.0, 3.0]),  # pesos de hiperaristas
    np.array([5.0, 7.0]),  # valores de hiperaristas
)
print(y)
```

`jit_cpu_aggregate()` devuelve un wrapper validado y usa Numba JIT cuando está instalado; `aggregate_incidence_numpy()` es la implementación de referencia. `emit_cpu_aggregate_source()` y `emit_cuda_aggregate_source()` generan el código fuente de esta operación. CUDA se puede consultar y ejecutar con `numba_cuda_available()` y `run_cuda_aggregate(...)`. Este proyecto **no** compila ni ejecuta código CUDA arbitrario desde HAL.

## Sintaxis HAL disponible

```text
f: A -> B                declaración y uso de morfismo
e1 # e2                  composición secuencial
e1 @ e2                  composición paralela
id(A)                    identidad
split(A)                 A -> A @ A
merge(A)                 A @ A -> A
cup(A)                   I -> A
cap(A)                   A -> I
swap(A, B)               A @ B -> B @ A
```

También se acepta `⊗` como alias de `@` en expresiones. Los comentarios de línea comienzan por `//` o `--`.

## Ejecutar y verificar

Desde el directorio del proyecto:

```powershell
python -m hal
python -m pytest -q
```

La suite actual tiene **31 pruebas** de parser, tipos, simplificación, matrices, tensores y generación/ejecución CPU. El smoke test de CLI también calcula la incidencia y la proyección. En el entorno en que se verificó este README, el JIT CPU pasó y Numba informó `CUDA device available: False`; no se afirmó ejecución GPU.

## Estructura

```text
hal/
  ast.py          AST
  parser.py       lexer y parser
  types.py        tipos monoidales
  typecheck.py    inferencia y unificación
  rewrite.py      reescrituras algebraicas orientadas
  hypergraph.py   incidencia, proyección y tensores
  kernels.py      backends de agregación y emisores
  __main__.py     demo: python -m hal
tests/
  test_hal.py
HAL_SPECIFICATION.md
```

## Limitaciones conocidas

- Las reescrituras cubren patrones algebraicos locales concretos; no son una forma normal completa ni un algoritmo general para decidir equivalencia de diagramas.
- El kernel JIT solo implementa la agregación CSR descrita arriba. No existe todavía un lowering general de AST HAL a kernels.
- No se implementan TACO, Tiga, composición de cospans por pushout, ni una arquitectura de compilación JIT C++/CUDA general.
- La ejecución CUDA necesita hardware/software externo compatible y no se pudo ejecutar en el entorno verificado.
- La representación tensorial requiere uniformidad; para hipergrafos heterogéneos utiliza incidencia.

La [especificación formal y técnica](HAL_SPECIFICATION.md) contiene los axiomas, reglas de tipado, convenciones de representación, dependencias y estado detallado de implementación.
