from Inicializaciones import *
from Graficaciones import *

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.sparse import csc_array

# configuracion de numpy que me permite ver la matriz jacobiana sin salto de linea
np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # se intenta evitar saltos de línea

# Definir dimensiones de la malla
Columnas, Filas = 50, 5

# Inicializar matriz de velocidad
vel_x_1 = Matriz_Redución_Completa(Filas, Columnas, 10)
vel_x_2 = Matriz_Redución_Eje_X(Filas, Columnas, 10)
vel_x_3 = Matriz_Velocidades_Uniforme(Filas, Columnas, 1)
vel_x_4 = Matriz_Velocidades_Uniforme(Filas, Columnas, 0)

# Función para calcular el vector F(X) evaluado en V
def construir_F(VX, vy, N, M, precision):

    F = np.zeros((M-2, N-2), dtype=float)
    
    for m in range (1, M-1):
        for n in range (1, N-1):
            eq = round((1/4) * (VX[m, n+1] + VX[m, n-1] + VX[m+1, n] + VX[m-1, n])
                       + (1/8) * (VX[m, n] * (VX[m-1, n]  - VX[m+1, n])
                       + vy * (VX[m, n-1] - VX[m, n+1])) - VX[m, n], precision)
            
            # Asignar el valor de la ecuación a la matriz F
            F[m-1, n-1] = eq

    return F.flatten(), F

# Función para calcular la matrix J(X) evaluada en V
def construir_J(VX, vy, N, M, constante, precision):
    total_cuadros_centro = (N - 2) * (M - 2)
    J = np.zeros((total_cuadros_centro, total_cuadros_centro), dtype=float)
    bordes_izq = []
    bordes_der = []
    
    for w in range(1, M - 3):
        bordes_izq.append((N-2) * w)
        bordes_der.append(((N-2) * (w+1)) - 1)

    for m in range(1, M - 1):
        for n in range(1, N - 1):

            k = (m - 1) * (N - 2) + (n - 1)
                
            #derivadas de la izquierda
            if k in bordes_izq:
                #J[k, k] = round(1/8 * (1 - VX[m, 2]) - constante, precision)
                J[k, k] = -1
                J[k, k+1] = round(1/4 - 1/8 * vy, precision)
                J[k, k+(N-2)] = round(1/4 - 1/8 * VX[m, 1], precision)
                J[k, k-(N-2)] = round(1/4 + 1/8 * VX[m, 1], precision)
                
            #derivadas de la derecha
            elif k in bordes_der:
                #J[k, k] = round(1/8 * VX[m, N-3] - constante, precision)
                J[k, k] = -1
                J[k, k-1] = round(1/4 + 1/8 * vy, precision)
                J[k, k+(N-2)] = round(1/4 - 1/8 * VX[m, N-2], precision)
                J[k, k-(N-2)] = round(1/4 + 1/8 * VX[m, N-2], precision)

            #derivadas del centro
            elif N - 1 <= k and k <= ((N - 2) * (M - 3)) - 2 and k not in bordes_izq and k not in bordes_der:
                #J[k, k] = round(1/8 * (VX[m, n-1] - VX[m, n+1]) - constante, precision)
                J[k, k] = -1
                J[k, k+1] = round(1/4 - 1/8 * vy, precision)
                J[k, k-1] = round(1/4 + 1/8 * vy, precision)
                J[k, k+(N-2)] = round(1/4 - 1/8 * VX[m, n], precision)
                J[k, k-(N-2)] = round(1/4 + 1/8 * VX[m, n], precision)
                    
            #derivadas de arriba
            elif 1 <= k and k <= (N - 4):
                #J[k, k] = round(1/8 * (VX[m, n-1] - VX[m, n+1]) - constante, precision)
                J[k, k] = -1
                J[k, k+1] = round(1/4 - 1/8 * vy, precision)
                J[k, k-1] = round(1/4 + 1/8 * vy, precision)
                J[k, k+(N-2)] = round(1/4 - 1/8 * VX[1, n], precision)
                
            #derivadas de abajo
            elif (((N - 2) * (M - 3)) + 1) <= k and k <= (((N - 2) * (M - 2)) - 2):
                #J[k, k] = round(1/8 * (VX[m, n-1] - VX[m, n+1]) - constante, precision)
                J[k, k] = -1
                J[k, k+1] = round(1/4 - 1/8 * vy, precision)
                J[k, k-1] = round(1/4 + 1/8 * vy, precision)
                J[k, k-(N-2)] = round(1/4 + 1/8 * VX[m, n], precision)
                
            #derivada cuadro esquina superior izquierda
            elif k == 0:
                #J[k, k] = round(1/8 * (1 - VX[1, 2]) - constante, precision)
                J[k, k] = -1
                J[k, k+1] = round(1/4 - 1/8 * vy, precision)
                J[k, k+(N-2)] = round(1/4 - 1/8 * VX[1, 1], precision)
                
            #derivada cuadro esquina superior derecha
            elif k == (N - 3):
                #J[k, k] = round(1/8 * VX[1, N - 3] - constante, precision)
                J[k, k] = -1
                J[k, k-1] = round(1/4 + 1/8 * vy, precision)
                J[k, k+(N-2)] = round(1/4 - 1/8 * VX[1, N-2], precision)
                
            #derivada cuadro esquina inferior izquierda
            elif k == ((N - 2) * (M - 3)):
                #J[k, k] = round(1/8 * (1 - VX[M-2, 2]) - constante, precision)
                J[k, k] = -1
                J[k, k+1] = round(1/4 - 1/8 * vy, precision)
                J[k, k-(N-2)] = round(1/4 + 1/8 * VX[M-2, 1], precision)

            #derivada cuadro esquina inferior derecha
            elif k == ((N - 2) * (M - 2) - 1):
                #J[k, k] = round(1/8 * VX[M-2, N-3] - constante, precision)
                J[k, k] = -1
                J[k, k-1] = round(1/4 + 1/8 * vy, precision)
                J[k, k-(N-2)] = round(1/4 + 1/8 * VX[M-2, N-2], precision)
            
            else:
                0

    return csc_array(J), J

# Método de Newton-Raphson
def resolver_sistema(VX, vy, c, N, M, tolerancia, ITER, ITER_OG, precision):
    try:
        Graficar_Matriz(VX, "Velocidad X", 'lower')
        print(f"Iteración {ITER_OG - ITER + 1}")

        F, S = construir_F(VX, vy, N, M, precision)
        J, K = construir_J(VX, vy, N, M, c, precision)
        H2 = np.linalg.solve(K, -F)
        H1 = spla.spsolve(J, -F)
        Graficar_Matriz(K, "J(X)", 'lower')
        Graficar_Matriz(F.reshape((M-2, N-2)), "F(X)", 'lower')
        max_H = max(abs(H1))

        #print("J", np.array2string(K, separator=', ', threshold=np.inf))
        #print("F", np.array2string(S, separator=', ', threshold=np.inf))
        #print("vx", VX)
        print("h1", H1, max_H)
        # print("h2", H2, max_H)
        # if (np.allclose(np.dot(K, H1), F)):
        #     print("La solución es exacta")
        # else:
        #     print("La solución no es exacta")
        # print(np.dot(K, H1))

        if max_H < tolerancia or ITER == 1:
            return VX
        
        nuevo_vx = VX
        nuevo_vx[1:M-1, 1:N-1] = VX[1:M-1, 1:N-1] + np.reshape(H1, (M-2, N-2))
        

        return resolver_sistema(nuevo_vx, vy, c, N, M, tolerancia, ITER-1, ITER_OG, precision)

    except spla.MatrixRankWarning:
        return VX
    
    except RuntimeError as e:
        return VX

# ValIniCentroMatrizVelX(Filas, Columnas, vel_x, 1, 10)
# distParabolica(Filas, Columnas, vel_x, 10)
i = 10
vel_x_final = resolver_sistema(vel_x_1, 0, 1, Columnas, Filas, 10e-4, i, i, 20)
Graficar_Matriz(vel_x_final, "Velocidad X final", 'lower')