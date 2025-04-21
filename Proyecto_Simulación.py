import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.sparse import csc_array
from Graficaciones import *
from Métodos_Resolver_Sistemas import *

# Definir dimensiones de la malla
N, M = 50, 8

# Inicializar matriz de velocidad
vel_x_1 = np.zeros((M, N))
vel_x_2 = np.zeros((M, N))
vel_x_3 = np.zeros((M, N))

vel_x_1[1:-1, 0] = 1  # Configurar la condicion inicial de la frontera
vel_x_2[1:-1, 0] = 1  # Configurar la condicion inicial de la frontera
vel_x_3[1:-1, 0] = 1  # Configurar la condicion inicial de la frontera

for m in range (1, M - 1):
    for n in range (1, N - 1):
        vel_x_1[m, n] = 1 - (n/(2*N-4)) #llenamos el centro con velocidades
        vel_x_2[m, n] = 1 #llenamos el centro con velocidades
        vel_x_3[m, n] = 0 #llenamos el centro con velocidades
        

# Función para calcular el vector F(X) evaluado en V
def construir_F(VX, vy, N, M):

    F = np.zeros((M-2, N-2), dtype=float)
    
    for n in range (1, N-1):
        for m in range (1, M-1):
            eq = (1/4) * (VX[m, n+1] + VX[m, n-1] + VX[m+1, n] + VX[m-1, n]
                + (1/2) * (-VX[m, n] * VX[m, n+1] + VX[m, n] * VX[m, n-1] 
                - vy * VX[m+1, n] + vy * VX[m-1, n]))
                
            # Asignar el valor de la ecuación a la matriz F
            F[m-1, n-1] = eq

    return F.flatten()

# Función para calcular la matrix J(X) evaluada en V
def construir_J(VX, vy, constante, N, M):
    total_cuadros_centro = (N - 2) * (M - 2)
    J = np.zeros((total_cuadros_centro, total_cuadros_centro), dtype=float)

    for n in range (total_cuadros_centro):
        for m in range (total_cuadros_centro):
            J[m, n] = 0

    for m in range(1, M - 1):
        for n in range(1, N - 1):

            k = (m - 1) * (N - 2) + (n - 1)
            
            #derivadas del centro
            if N - 1 <= k <= ((N - 2) * (M - 3)) - 2:
                J[k, k] = 1/8 * VX[m, n-1] - 1/8 * VX[m, n+1] - constante
                J[k, k+1] = 1/4 - 1/8 * VX[m, n]
                J[k, k-1] = 1/4 + 1/8 * VX[m, n]
                J[k, k+(N-2)] = 1/4 - 1/8 * vy
                J[k, k-(N-2)] = 1/4 + 1/8 * vy
                
            #derivadas de la izquierda
            if k % (N - 2) == 0 and k != 0 and k != (N - 2) * (M - 3):
                J[k, k] = 1/8 - 1/8 * VX[m, 2] - constante
                J[k, k+1] = 1/4 - 1/8 * VX[m, 1]
                J[k, k+(N-2)] = 1/4 - 1/8 * vy
                J[k, k-(N-2)] = 1/4 + 1/8 * vy
                
            #derivadas de la derecha
            for w in range(2, M - 2):
                if (k / (w * (N - 2) - 1)) == 1 and k != 0:
                    J[k, k] = 1/8 * VX[m, N-3] - constante
                    J[k, k-1] = 1/4 + 1/8 * VX[m, N-2]
                    J[k, k+(N-2)] = 1/4 - 1/8 * vy
                    J[k, k-(N-2)] = 1/4 + 1/8 * vy
                    
            #derivadas de arriba
            if 1 <= k <= (N - 4):
                J[k, k] = 1/8 * VX[m, n-1] - 1/8 * VX[m, n+1] - constante
                J[k, k+1] = 1/4 - 1/8 * VX[1, n]
                J[k, k-1] = 1/4 + 1/8 * VX[1, n]
                J[k, k+(N-2)] = 1/4 - 1/8 * vy
                
            #derivadas de abajo
            elif (((N - 2) * (M - 3)) + 1) <= k <= (((N - 2) * (M - 2)) - 2):
                J[k, k] = 1/8 * VX[m, n-1] - 1/8 * VX[m, n+1] - constante
                J[k, k+1] = 1/4 - 1/8 * VX[m, n]
                J[k, k-1] = 1/4 + 1/8 * VX[m, n]
                J[k, k-(N-2)] = 1/4 + 1/8 * vy
                
            #derivada cuadro esquina superior izquierda
            elif k == 0:
                J[k, k] = 1/8 - 1/8 * VX[1, 2] - constante
                J[k, k+1] = 1/4 - 1/8 * VX[1, 1]
                J[k, k+(N-2)] = 1/4 - 1/8 * vy
                
            #derivada cuadro esquina superior derecha
            elif k == (N - 3):
                J[k, k] = 1/8 * VX[1, N - 3] - constante
                J[k, k-1] = 1/4 + 1/8 * VX[1, N-2]
                J[k, k+(N-2)] = 1/4 - 1/8 * vy
                
            #derivada cuadro esquina inferior izquierda
            elif k == ((N - 2) * (M - 3)):
                J[k, k] = 1/8 - 1/8 * VX[M-2, 2] - constante
                J[k, k+1] = 1/4 - 1/8 * VX[M-2, 1]
                J[k, k-(N-2)] = 1/4 + 1/8 * vy

            #derivada cuadro esquina inferior derecha
            elif k == ((N - 2) * (M - 2) - 1):
                J[k, k] = 1/8 * VX[M-2, N-3] - constante
                J[k, k-1] = 1/4 + 1/8 * VX[M-2, N-2]
                J[k, k-(N-2)] = 1/4 + 1/8 * vy
                
            else:
                0

    return csc_array(J), J

# Método de Newton-Raphson
def resolver_sistema(VX, vy, c, N, M, tolerancia, ITER):
    try:
        #print(f"Iteración {ITER_OG - ITER + 1}")
        F = construir_F(VX, vy, N, M)
        J, K = construir_J(VX, vy, c, N, M)

        print("\nMetodo del gradiente descendente")
        i1, h1 = gradiente_descendente(K, VX, F, N, M, ITER)
        print(i1, h1)

        print("\nMetodo del gradiente conjugado")
        i2, h2 = gradiente_conjugado(K, VX, F, N, M, ITER, 1e-8, 1e-8)
        print(i2, h2)

        Graficar_Jacobiano(K)
        #H = spla.spsolve(J, -F)

        #max_H = max(abs(H))
        #print(H, max_H)
        
        #if max_H < tolerancia or ITER == 0:
        #    return VX, ITER
        H1 = VX
        H2 = VX
        
        H1 = H1[1:M-1, 1:N-1] + h1.reshape((M-2, N-2))
        H2 = H2[1:M-1, 1:N-1] + h2.reshape((M-2, N-2))
        
        return i1, H1, i2, H2

    except spla.MatrixRankWarning:
        return VX, ITER
    
    except RuntimeError as e:
        return VX, ITER

# Mostrar resultados
def graficar_resultado(vx_final, iter_final, c):
    plt.figure(figsize=(12, 6))
    plt.imshow(vx_final, cmap='jet', interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(label='v_x')
    plt.title(f'Distribución final de v_x (c={c}, Iteración {iter_final})', fontsize=14)
    plt.xlabel('Índice i', fontsize=12)
    plt.ylabel('Índice j', fontsize=12)
    plt.show()

vx_i_1, vx_final_1, vx_i_2, vx_final_2 = resolver_sistema(vel_x_3, 0, -1, N, M, 1, 8)

graficar_resultado(vx_final_1, vx_i_1, -1)
graficar_resultado(vx_final_2, vx_i_2, -1)

# for c_val in [0, 1, -1]:
#     vx_final, iter_final = resolver_sistema(vel_x_1, vel_x_1, 1, c_val, N, M, 1, 2, 2)
#     graficar_resultado(vx_final, iter_final, c_val)

# for c_val in [0, 1, -1]:
#     vx_final, iter_final = resolver_sistema(vel_x_2, vel_x_2, 0, c_val, N, M, 1, 2, 2)
#     graficar_resultado(vx_final, iter_final, c_val)

# for c_val in [0, 1, -1]:
#     vx_final, iter_final = resolver_sistema(vel_x_3, vel_x_3, 0, c_val, N, M, 1, 2, 2)
#     graficar_resultado(vx_final, iter_final, c_val)