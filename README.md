# Proyecto Final Redes Neuronales 2021-1

## Propuesta de Control con Redes Neuronales

Alma Rocío Sánchez Salgado 
Adrián Bruce Nolasco Cabello 

Para el proyecto final del curso proponemos el uso conjunto de dos redes neuronales artificiales para el control de sistemas dinámicos sencillos. Una de las redes , que denominamos _navegante_, modela el sistema dinámico.

La segunda red establece las condiciones de control tomando como entradas las condiciones actuales del sistema y el pronóstico de la red navegante. Si la red navegante es entrenada de forma satisfactoria la salida permite definir una función de error con la cual se puede entrenar a la red _piloto_ sin necesidad de tomar datos directamente del sistema.

En nuestro caso utilizaremos simulaciones de sistemas mecánicos sencillos que simularemos mediante integración numérica de las ecuaciones de movimiento. Las series de tiempo de estas simulaciones las utilizaremos para entrenar la red navegante y después utilizaremos la red navegante en conjunto con la red piloto para hacer rondas de entrenamiento con el sistema simulado dónde la función de error para la red piloto estará dada por la diferencia entre las condiciones deseadas y las actuales.
