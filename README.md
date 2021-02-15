# Proyecto Final Redes Neuronales 2021-1
# Propuesta de Control con Redes Neuronales

- Alma Rocío Sánchez Salgado
- Adrián Bruce Nolasco Cabello

Para el proyecto final del curso proponemos el uso conjunto de dos redes neuronales artificiales para la simulación y control de una grúa.

### Introducción

Una de las redes , que denominamos _navegante_, modela el sistema dinámico mientras que la segunda red establece las condiciones de control tomando como entradas las condiciones actuales del sistema. Si la red navegante es entrenada de forma satisfactoria, la salida permitirá definir una función de error con la cual se puede entrenar a la red _piloto_ sin necesidad de tomar datos directamente del sistema.

Para diseñar una política de control que haga que el sistema se comporte de manera deseada, necesitamos predecir el comportamiento de las variables de interés a través del tiempo, específicamente, cómo cambian en repuesta a diferentes entradas. Por lo anterior, obtendremos las series de tiempo a partir de simulaciones de sistemas mecánicos mediante integración númerica de las ecuaciones de movimiento. Después utilizaremos la red navegante en conjunto con la red piloto para hacer rondas de entrenamiento con el sistema simulado donde la función de error para la red piloto estará dada por la diferencia entre las condiciones deseadas y las actuales.

Este método permitiría controlar sistemas de los cuales tenemos información limitada y/o de aquellos en donde resulta muy costoso obtener datos a través de experimentación directa ya que, mediante su modelo podremos entrenar el _piloto_ con la simulación del sistema y obtener los datos necesarios para después usarlos para el entrenamiento de la red piloto.

Dado lo anterior, las fuentes de datos serían las simulaciones del sistema. Los programas de estas simulaciones se subirán a este mismo repositorio.

### Objetivos

El objetivo principal es entrenar satisfactoriamente la red navagante para que reproduzca la dinámica de la grúa.

Si esto se logra entonces sería posible proceder con la parte de control del sistema y a entrenar la red piloto.

### Conjunto de Datos

La dinámica de la grúa está descrita por la siguiente ecuación:
 
$$ l\ddot{\theta} = -2\dot{l} \dot{\theta}-g\sin \theta -m \ddot{u} \cos \theta $$ 
 
El estado del sistema consiste de 6 variables la posición $u$ y velocidad del carro de la grúa, la longitud $l$ y velocidad de retracción del cable de la grúa y la inclinación $\theta$ y velocidad angular de la carga.

De estas 6 variables asumimos que podemos controlar 2 mediante motores: la velocidad del carro y la velocidad de retracción. Consideramos que la grúa tiene una velocidad máxima de retracción y desplazamiento. Además dada la inercia mecánica del sistema estas señales de control pasan por un filtro pasabajos.

Mediante los métodos de integración para ecuaciones diferenciales incluidos en el modulo _ scipy _ y _ numpy _ es posible simular la evolución del sistema en respuesta una señal arbitraria para las 2 variables de control mencionadas. 

Cada caso consistirá de 6 series de tiempo (una para cada variable). De estas, las correspondientes a las velocidades serán 2 señales aleatorias generadas mediante una caminata aleatoria gaussiana dentro del rango de velocidades máximas para la grúa. 

Muestreamos esta serie en un número determinado de tiempos distintos y almacenamos el conjunto de muestras con los tiempos correspondientes y las 2 señales de control.

## Diseño de las redes

El objetivo de la red navegante es obtener la función tal que:
 
$$ \frac{d}{dt} \mathbf{s} =  \mathbf{f}(\mathbf{s},\mathbf{z},\theta) $$

a partir de las series de tiempo y las señales de control proporcionadas en cada caso. La red navegante tiene la arquitectura _ NeuralODE _ descrita en [2].

Por otra parte el objetivo de la red piloto es determinar en cada instante en función del estado de la grúa y el estado objetivo las señales de control indicadas para alcanzar el estado objetivo. 

La red piloto es una red perceptrón multicapa. La capa de salida tiene 2 neuronas de activación arcotangente hiperbólica de tal forma que un -1 se interpreta como velocidad máxima de retracción o retroceso y un 1 como velocidad máxima de elongación o avance.

### Entrenamiento

Para el caso de la red navegante el error estará dado por la función MSE para la diferencia entre el estado predicho al tiempo t_i y la observación al tiempo t_i partiendo del estado inicial en t=0 y con las señales de control de cada caso.

El entrenamiento de una red _ NeuralODE _ para ajustar una dinámica a partir de series de tiempo se describe en [2] y una implementación en _Torch_ se encuentra en https://github.com/rtqichen/torchdiffeq/. 

[1] Javier E Vitela & Julio J Martinell. Stabilization of burn conditions in a thermonuclear reactorusing artificial neural networks. Plasma Phys. Control. Fusion40(1998) 295–318.  http://www.nucleares.unam.mx/~martinel/ftp/netiter1.pdf

[2] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt & David Duvenaud. Neural Ordinary Differential Equations. https://arxiv.org/pdf/1806.07366.pdf
