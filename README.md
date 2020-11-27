# Proyecto Final Redes Neuronales 2021-1

## Propuesta de Control con Redes Neuronales

Alma Rocío Sánchez Salgado 
Adrián Bruce Nolasco Cabello 

Para el proyecto final del curso proponemos el uso conjunto de dos redes neuronales artificiales para el control de sistemas dinámicos sencillos.

Una de las redes , que denominamos _navegante_, modela el sistema dinámico mientras que la segunda red establece las condiciones de control tomando como entradas las condiciones actuales del sistema y el pronóstico de la red navegante. Si la red navegante es entrenada de forma satisfactoria, la salida permitirá definir una función de error con la cual se puede entrenar a la red _piloto_ sin necesidad de tomar datos directamente del sistema.

En nuestro caso utilizaremos simulaciones de sistemas mecánicos sencillos que simularemos mediante integración numérica de las ecuaciones de movimiento. Las series de tiempo de estas simulaciones las utilizaremos para entrenar la red navegante y después utilizaremos la red navegante en conjunto con la red piloto para hacer rondas de entrenamiento con el sistema simulado donde la función de error para la red piloto estará dada por la diferencia entre las condiciones deseadas y las actuales.

Consideramos que es interesante resolver este problema pues este método permitiría controlar el sistema con información limitada (las medidas previas para entrenar el _navegante_), construir un modelo del mismo y poder entrenar el _piloto_ en simulación sin necesidad de realizar experimentos directamente sobre el sistema físico.

Dado lo anterior, las fuentes de datos serían las simulaciones del sistema. Los programas de estas simulaciones se subirán a este mismo repositorio. Aún no definimos los sistemaa dinámicoa que deseamos trabajar pues no deseamos que sea tan sencillo pero si requerimos un modelo del mismo. Un ejemplo de aplicación de una técnica de control con redes neuronales se halla en [1]. Un área de oportunidad es implementar una arquitectura de red novedosa mostrada en [2] para el _navegante_. 

[1] Javier E Vitela & Julio J Martinell. Stabilization of burn conditions in a thermonuclear reactorusing artificial neural networks. Plasma Phys. Control. Fusion40(1998) 295–318.  http://www.nucleares.unam.mx/~martinel/ftp/netiter1.pdf

[2] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt & David Duvenaud. Neural Ordinary Differential Equations. https://arxiv.org/pdf/1806.07366.pdf
