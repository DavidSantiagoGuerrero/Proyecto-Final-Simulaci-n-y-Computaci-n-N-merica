import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Definir dimensiones de la malla
N, M = 50, 8

# Inicializar matriz de velocidad
vel_x = np.zeros((M, N))

vel_x[1:-1, 0] = 1  # Configurar la condicion inicial de la frontera

for m in range (1, M - 1):
    for n in range (1, N - 1):
        vel_x[m, n] = 1 - (n/(N-2)) #llenamos el centro con velocidades

# Función para calcular el vector F(X) evaluado en V

def construir_F(VX, vy, N, M):

    F = np.zeros((M-2, N-2))
    
    for n in range (1, N-1):
        for m in range (1, M-1):

            #ecuación de la linea de arriba
            if m == 1 and 2 <= n <= N - 3:
                eq = (1/4)*(VX[1, n+1] + VX[1, n-1] + VX[2, n]
                    + 1/2 * (-VX[m, n] * VX[1, n+1] + VX[m, n] * VX[1, n-1] - vy * VX[2, n]))
                
            #ecuación de la linea de abajo
            elif m == M - 2 and 2 <= n <= N - 3:
                eq = (1/4)*(VX[M-2, n+1] + VX[M-2, n-1] + VX[M-3, n]
                    + 1/2 * (-VX[m, n] * VX[M-2, n+1] + VX[m, n] * VX[M-2, n-1] + vy * VX[M-3, n]))
                
            #ecuacion de la linea de la izquierda
            elif n == 1 and 2 <= m <= M - 3:
                eq = (1/4)*(VX[m, 2] + 1 + VX[m+1, 1] + VX[m-1, 1]
                    + 1/2 * (-VX[m, n] * VX[m, 2] + VX[m, n] - vy * VX[m+1, 1] + vy * VX[m-1, 1]))
                
            #ecuacion de la linea de la derecha
            elif n == N - 2 and 2 <= n <= M - 3:
                eq = (1/4)*(VX[m, N-3] + VX[m+1, N-2] + VX[m-1, N-2]
                    + 1/2 * (VX[m, n] * VX[m, N-3] - vy * VX[m+1, N-2] + vy * VX[m-1, N-2]))
                
            #ecuacion de la esquina superior izquierda
            elif (n, m) == (1, 1):
                eq = (1/4)*(VX[1, 2] + 1 + VX[2, 1]
                    + 1/2 * (-VX[m, n] * VX[1, 2] + VX[m, n] - vy * VX[2, 1]))
                
            #ecuacion de la esquina inferior izquierda
            elif (n, m) == (1, M-2):
                eq = (1/4)*(VX[M-2, 2] + 1 + VX[M-3, 1]
                    + 1/2 * (-VX[m, n] * VX[M-2, 2] + VX[m, n] + vy * VX[M-3, 1]))
                
            #ecuacion de la esquina superior derecha
            elif (n, m) == (N-2, 1):
                eq = (1/4)*(VX[1, N-3] + VX[2, N-2]
                    + 1/2 * (VX[m, n] * VX[1, N-3] - vy * VX[2, N-2]))
                
            #ecuacion de la esquina inferior derecha
            elif (n, m) == (N-2, M-2):
                eq = (1/4)*(VX[M-2, N-3] + VX[M-3, N-2]
                    + 1/2 * (VX[m, n] * VX[M-2, N-3] + vy * VX[M-3, N-2]))
                
            #ecuacion del centro
            else:
                eq = (1/4)*(VX[m, n+1] + VX[m, n-1] + VX[m+1, n] + VX[m-1, n]
                    + 1/2*(-VX[m, n] * VX[m, n+1] + VX[m, n] * VX[m, n-1] - vy * VX[m+1, n] + vy * VX[m-1, n]))
                
            # Asignar el valor de la ecuación a la matriz F
            F[m-1, n-1] = eq

    return F.flatten()

#F = construir_F(vel_x, 0, N, M)
#print(F.shape)
# plt.imshow(F, cmap='hot', interpolation='nearest')
# plt.colorbar(label="Valor de F")
# plt.title("Mapa de calor de F")
# plt.xlabel("Eje X")
# plt.ylabel("Eje Y")
# plt.show()

# Función para calcular la matrix J(X) evaluada en V
def construir_J(VX, vy, N, M):
    total_cuadros_centro = (N - 2) * (M - 2)
    J = sp.lil_matrix((total_cuadros_centro, total_cuadros_centro))
    constante = 1

    for n in range (total_cuadros_centro):
        for m in range (total_cuadros_centro):
            J[n, m] = 0

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

    return J.tocsr()

# Método de Newton-Raphson
def resolver_sistema(vx, vy, N, M, tolerancia, ITER, ITER_OG):
    try:
        print(f"Iteración {ITER_OG - ITER + 1}")

        F = construir_F(vx, vy, N, M)
        J = construir_J(vx, vy, N, M)
        H = spla.spsolve(J, -F)

        max_H = max(abs(H))
        print(H, max_H)

        if max_H < tolerancia or ITER == 0:
            return vx
        
        nuevo_vx = vx
        nuevo_vx[1:M-1, 1:N-1] = vx[1:M-1, 1:N-1] + H.reshape((M-2, N-2))

        return resolver_sistema(nuevo_vx, vy, N, M, tolerancia, ITER-1, ITER_OG)

    except spla.MatrixRankWarning:
        return vx
    
    except RuntimeError as e:
        return vx

# Mostrar resultados
def graficar_resultado(vx_final):
    plt.figure(figsize=(12, 6))
    plt.imshow(vx_final, cmap='jet', interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(label='v_x')
    plt.title('Distribución final de v_x', fontsize=14)
    plt.xlabel('Índice i', fontsize=12)
    plt.ylabel('Índice j', fontsize=12)
    plt.show()

vel_x_final = resolver_sistema(vel_x, 0, N, M, 1, 1, 1)
graficar_resultado(vel_x_final)