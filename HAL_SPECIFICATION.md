# HAL: formalización algebraica, semántica e implementación

**HAL (Hyperalgebra Language)** es un lenguaje pequeño para describir composiciones de morfismos y estructuras de hipergrafos. Este repositorio contiene un prototipo Python modular con parser, comprobación de tipos, reescrituras algebraicas, representaciones NumPy/SciPy y kernels de agregación CPU/CUDA.

> Alcance honesto: este código es un prototipo ejecutable de investigación, no un compilador industrial ni una implementación del word problem completo de la categoría hipergrafo libre. Las partes implementadas, opcionales y aún no implementadas se distinguen explícitamente abajo.

## 1. Semántica categórica

Los programas se interpretan en una categoría monoidal simétrica estricta
`(H, @, I, swap)`. Los tipos son objetos; las expresiones bien tipadas son morfismos. `@` es el producto tensorial y `I` su unidad. En la implementación, los productos se aplanan y `I @ X = X = X @ I`; esto fija asociadores y unitores a una representación canónica.

Un morfismo `f: A -> B` tiene dominio `A` y codominio `B`. El cospan decorado `X_in -> A <- X_out` es una interpretación topológica útil, pero **el prototipo aún no implementa una estructura `Cospan` ni construye pushouts**.

Por objeto `X` se postula un álgebra de Frobenius conmutativa especial:

| Primitiva | HAL | Tipo |
|---|---|---|
| Coproducto | `split(X)` | `X -> X @ X` |
| Producto | `merge(X)` | `X @ X -> X` |
| Unidad | `cup(X)` | `I -> X` |
| Counidad | `cap(X)` | `X -> I` |
| Simetría | `swap(X, Y)` | `X @ Y -> Y @ X` |

La secuenciación en HAL se escribe de izquierda a derecha: `e1 # e2` denota `⟦e2⟧ ∘ ⟦e1⟧`. El tensor denota `⟦e1⟧ @ ⟦e2⟧`.

### Axiomas expresados en orden HAL

`id(X)` es la identidad. Las ecuaciones siguientes son las que informa `hal.rewrite.RULES` y el reescritor aplica sobre patrones estructurales tipados:

* Asociatividad: `(merge @ id)#merge = (id @ merge)#merge`.
* Coasociatividad: `split#(split @ id) = split#(id @ split)`.
* Conmutatividad: `swap(X,X)#merge(X) = merge(X)` y `split(X)#swap(X,X) = split(X)`.
* Frobenius: `(id @ split)#(merge @ id) = merge#split = (split @ id)#(id @ merge)`.
* Especialidad: `split#merge = id`.
* Unidad: `(cup @ id)#merge = id` y `(id @ cup)#merge = id`.
* Counidad: `split#(cap @ id) = id` y `split#(id @ cap) = id`.
* Involución de simetría: `swap(A,B)#swap(B,A) = id(A @ B)`.
* Identidad secuencial y tensorial: `id#f=f`, `f#id=f`, `id(A)@id(B)=id(A@B)`.

Estas son reglas de reescritura orientadas para normalizar los patrones reconocidos. La igualdad categórica es congruencial y bidireccional; por ello, el normalizador **no afirma** decidir toda equivalencia posible ni produce una forma normal completa para cualquier diagrama.

## 2. Sintaxis y parser

### Gramática práctica

```ebnf
expression   ::= parallel { "#" parallel } ;
parallel     ::= atom { ("@" | "⊗") atom } ;
atom         ::= identifier
               | identifier ":" type "->" type
               | "id(" type ")"
               | "split(" type ")"
               | "merge(" type ")"
               | "cup(" type ")"
               | "cap(" type ")"
               | "swap(" type "," type ")"
               | "(" expression ")" ;
type         ::= "I" | identifier | "(" type ("@" type)+ ")" ;
```

`@`/`⊗` tiene precedencia mayor que `#`; ambos son asociativos a izquierda en el AST. Los comentarios de línea empiezan por `//` o `--`. Los tipos escritos con inicial mayúscula (por ejemplo `Node`) son tipos base; identificadores de tipo en minúscula (por ejemplo `a`) son variables de tipo polimórficas. Para referencias a morfismos se proporciona un registro de firmas, o se declara el morfismo en la expresión: `f: A -> B # f`.

El parser informa de posiciones sintácticas y **rechaza** morfismos sin firma; no inventa dominios/codominios para entradas desconocidas.

```python
from hal import BaseType, parse_hal, infer_type

signatures = {
    "f": (BaseType("A"), BaseType("B")),
    "g": (BaseType("C"), BaseType("D")),
}
term = parse_hal("f @ g", signatures)
print(infer_type(term))
```

El ejemplo infiere `(A @ C) -> (B @ D)`. Para usar `#`, el codominio del morfismo izquierdo debe coincidir con el dominio del derecho.

## 3. Inferencia y comprobación de tipos

`hal.typecheck.TypeChecker` implementa unificador con sustituciones y comprobación de occurs-check. Cada uso de una firma polimórfica freshena sus variables de tipo, evitando que llamadas distintas compartan accidentalmente restricciones. Se comprueban:

* compatibilidad del codominio/dominio en `#`,
* dominios/codominios tensoriales en `@`,
* aridad y estructura de productos,
* resolución de variables y detección de tipos recursivos.

`infer_type(expr)` devuelve el par `(dominio, codominio)` ya resuelto o lanza `TypeCheckError`. Las firmas que llegan del registro son `(Type, Type)`; se puede expresar polimorfismo creando `TypeVariable("a")` y reutilizando esa variable en dominio y codominio.

Formalmente, `Γ` es el entorno de morfismos primitivos:

```text
Γ ⊢ f : A -> B
────────────────────
Γ ⊢ f : A -> B

────────────────────
Γ ⊢ id(A) : A -> A

Γ ⊢ e1 : A -> B      Γ ⊢ e2 : B -> C
────────────────────────────────────
Γ ⊢ e1 # e2 : A -> C

Γ ⊢ e1 : A -> B      Γ ⊢ e2 : C -> D
────────────────────────────────────
Γ ⊢ e1 @ e2 : A @ C -> B @ D
```

Las reglas de `split`, `merge`, `cup`, `cap` y `swap` son exactamente sus tipos de la tabla categórica de la sección 1. En el caso polimórfico, cada instanciación de `Γ(f)` sustituye sus variables por variables frescas antes de unificar.

## 4. Reescrituras algebraicas

`hal.rewrite.rewrite(expr, max_steps=10_000)` valida primero la expresión y después:

1. elimina identidades secuenciales,
2. canoniza la asociación de secuencias y productos,
3. aplica las leyes especial, unidad, counidad y conmutatividad,
4. canoniza las dos variantes de asociatividad/coasociatividad,
5. orienta las dos leyes de Frobenius hacia `merge#split`,
6. simplifica el producto de identidades y la involución de `swap`.

La medida `max_steps` limita el trabajo y los patrones están orientados para evitar oscilaciones. Los tests comparan formas resultantes, además de comprobar el tipado antes/después. El conjunto **no** reescribe diagramas arbitrarios bajo todos los contextos de la categoría ni cubre exhaustivamente la coherencia simétrica (por ejemplo, cualquier permutación de muchos cables); no se debe usar como demostrador general de equivalencia.

## 5. Backend de hipergrafos

`Hypergraph(vertices, edges)` valida nombres únicos, referencias a vértices existentes, ausencia de vértices repetidos en una hiperarista y pesos finitos.

### Incidencia y proyección de clique

`incidence_matrix(format="dense" | "csr" | "auto")` produce `B ∈ R^(|V| × |E|)`. Dense y cálculo tensorial usan NumPy. `csr` requiere SciPy; si no está instalado se informa el error, no se degrada silenciosamente.

`clique_projection()` devuelve:

* `gram = B W B.T`,
* `degree = diag(diag(gram))`,
* `adjacency = gram - degree`, por tanto adyacencia proyectada sin diagonal y `gram = adjacency + degree`.

Se conservan pesos de hiperaristas. `normalize=True` escala las entradas fuera de diagonal usando los grados diagonales de `gram`. La proyección densa requiere espacio cuadrático; CSR se mantiene dispersa cuando la incidencia se pide como CSR.

Si `H=(V,E)` tiene `N=|V|` y `K=|E|`, la matriz es `N×K` y almacena `Σ_e |e|` unos (más sobrecarga de índices en CSR), en vez de un tensor denso `N^M`. Para una hiperarista ponderada simple, `B W B.T` contiene `w_e` en cada par de vértices co-incidentes y `|e| w_e` en la diagonal correspondiente a sus nodos. La diagonal se conserva como `degree`; se separa de la adyacencia proyectada para que se cumpla la identidad documentada sin ambigüedad.

### Tensor uniforme

`adjacency_tensor(format="dense" | "coo")` está restringido a hipergrafos uniformes. Es completamente simétrico en sus índices y acumula pesos de hiperaristas repetidas. `coo` devuelve `SparseTensorCOO` (coordenadas, valores y forma) y limita por defecto la expansión simétrica a un millón de entradas; `dense` limita por defecto la asignación a 10 millones de elementos y lanza `MemoryError` si excede el umbral. No se materializa tensor denso para una estructura heterogénea.

`tensor_contraction(signals)` implementa una agregación simétrica uniforme sencilla en NumPy/Python. No es un optimizador TACO ni una contracción general de expresiones HAL.

## 6. Kernels CPU y GPU

El kernel actualmente definido es agregación dispersa de datos reales de hiperarista a vértice, sobre incidencia CSR:

`y[v] = Σ_e B[v,e] * edge_weights[e] * edge_values[e]`.

* `aggregate_incidence_numpy(...)`: referencia NumPy/Python validada.
* `jit_cpu_aggregate()`: devuelve un wrapper validado sobre el loop compilado Numba CPU si Numba está instalado; en otro caso devuelve la referencia Python.
* `emit_cpu_aggregate_source()`: emite código Python ejecutable para el mismo cálculo.
* `get_cuda_aggregate_kernel()` y `run_cuda_aggregate(...)`: obtienen/lanzan un kernel Numba CUDA; la ejecución requiere Numba y dispositivo/runtime CUDA disponible.
* `emit_cuda_aggregate_source()`: genera fuente CUDA C++ que puede compilarse con una toolchain CUDA externa.

La prueba ejecuta efectivamente la ruta CPU y compila/ejecuta el fuente Python generado. La GPU depende del entorno: `numba_cuda_available()` permite consultar dispositivo/runtime y `run_cuda_aggregate` falla explícitamente si no hay CUDA. Este repositorio no incluye un compilador NVCC, instalación CUDA ni kernels para operaciones HAL arbitrarias.

## 7. Estructura del proyecto

```text
hal/
  ast.py          nodos de AST
  parser.py       lexer y parser
  types.py        tipos base, productos, unidad y variables
  typecheck.py    unificador/inferencia
  rewrite.py      normalizador algebraico orientado
  hypergraph.py   hipergrafo, incidencia, proyección, tensor
  kernels.py      agregación NumPy/Numba/CUDA y emisores
  __main__.py     demostración ejecutable
tests/
  test_hal.py     pruebas unitarias de los componentes anteriores
```

## 8. Instalación, pruebas y demo

Python 3.10 o posterior.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[all]"
python -m pytest -q
python -m hal
```

`pip install -r requirements.txt` instala NumPy y SciPy. `pip install -r requirements-jit.txt` añade Numba. El extra `.[all]` añade Numba y pytest.

## 9. Módulos y funciones públicas

Las interfaces principales se exportan desde `hal`:

```python
parse_hal, infer_type, rewrite
Hypergraph, Hyperedge
aggregate_incidence_numpy, jit_cpu_aggregate, run_cuda_aggregate
```

El CLI `python -m hal` ejecuta una expresión tipada, crea un hipergrafo heterogéneo, calcula su proyección CSR/densa y ejecuta el agregado CPU.

## 10. Estado de alcance frente a la propuesta inicial

| Capacidad solicitada | Estado real |
|---|---|
| Parser real | Implementado: tokens con offsets, precedencia, tipos, errores sintácticos, declaraciones inline y firmas externas. |
| Inferencia de tipos | Implementada: firma polimórfica freshen, unificación, occurs-check y validación secuencial/paralela. |
| Reescrituras de Frobenius | Implementadas para las identidades enumeradas y patrones locales exactos; no es un decision procedure completo para la categoría libre. |
| Incidencia NumPy/SciPy | Implementada: matriz dense/CSR, pesos, proyección `BWB.T`, grado, adyacencia y normalización. SciPy es dependencia declarada. |
| Tensor uniforme | Implementado: tensor simétrico dense acotado o COO disperso y una contracción de señales de referencia. |
| Kernels JIT CPU/GPU | CPU Numba y fuente CPU probados; CUDA Numba/source implementados, pero la ejecución real necesita una GPU/runtime CUDA. |
| Compilación de cualquier AST HAL a kernel | No implementada; el kernel actual cubre agregación CSR de hiperarista a vértice. |
| TACO/Tiga, cospans/pushouts y emisión general C++/CUDA | No implementados en esta versión. |

Los tests son la fuente de verdad sobre el comportamiento actual. Para extender hacia un compilador completo, los siguientes hitos son: semántica de cospans y composición por pushout, normalización completa con coherencia simétrica, planner con estimación de densidad/orden y lowering de AST a kernels.
