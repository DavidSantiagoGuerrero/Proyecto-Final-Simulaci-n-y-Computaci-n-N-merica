import numpy as np

#funcion que llena la matriz con un valor uniforme
def Matriz_Velocidades_Uniforme(filas, columnas, velocidad):
    Matriz = np.zeros((filas, columnas), dtype=float)
    Matriz[1:-1, 0] = 1

    for m in range(1, filas - 1):
        for n in range(1, columnas - 1):
            Matriz[m, n] = velocidad

    return Matriz

#funcion que llena la matriz con velocidades reduciendose uniformemente en el eje x
def Matriz_Redución_Eje_X(filas, columnas, precision):
    Matriz = np.zeros((filas, columnas), dtype=float)
    Matriz[1:-1, 0] = 1

    for m in range (1, filas - 1):
        for n in range (1, columnas - 1):
            Matriz[m, n] = round(1 - (n/(columnas-1)), precision)

    return Matriz

#funcion que llena la matriz con velocidades reduciendose uniformemente en el eje x e y
def Matriz_Redución_Completa(filas, columnas, precision):
    Matriz = Matriz_Redución_Eje_X(filas, columnas, precision)
    fil = int(filas/2)
    
    if (filas % 2 == 0):
        for m in range (1, int(fil)):
            factor = 1 - abs(fil - m) / fil  
            for n in range (1, columnas - 1):
                velocidad = round(Matriz[m, n] * factor, precision)

                Matriz[m, n] = velocidad
                Matriz[filas - (m + 1), n] = velocidad
    else:
        for m in range (1, int(fil) + 1):
            factor = 1 - abs(fil - m) / fil
            for n in range (1, columnas - 1):
                velocidad = round(Matriz[m, n] * factor, precision)

                if (m != int(fil)):
                    Matriz[m, n] = velocidad
                    Matriz[filas - (m + 1), n] = velocidad
                else:
                    Matriz[m, n] = velocidad
    return Matriz