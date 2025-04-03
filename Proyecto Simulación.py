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
        vel_x[m, n] = 1 #llenamos el centro con velocidades

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
    casos = np.zeros((total_cuadros_centro, total_cuadros_centro))  # Matriz para identificar los casos

    constante = 0

    for n in range (total_cuadros_centro):
        for m in range (total_cuadros_centro):
            J[n, m] = 0

    for m in range(1, M - 1):
        for n in range(1, N - 1):

            k = (m - 1) * (N - 2) + (n - 1)
            
            #derivadas del centro
            if N - 1 <= k <= ((N - 2) * (M - 3)) - 2:
                J[k, k] = 1/8 * VX[m, n-1] - 1/8 * VX[m, n+1] - constante
                casos[k, k] = 5

                J[k, k+1] = 1/4 - 1/8 * VX[m, n]
                casos[k, k+1] = 1

                J[k, k-1] = 1/4 + 1/8 * VX[m, n]
                casos[k, k-1] = 2

                J[k, k+(N-2)] = 1/4 - 1/8 * vy
                casos[k, k+(N-2)] = 3

                J[k, k-(N-2)] = 1/4 + 1/8 * vy
                casos[k, k-(N-2)] = 4

                print((k, k), (k, k+1), (k, k-1), (k, k+(N-2)), (k, k-(N-2)), (n-1, m), (n+1, m), (n, m))

            #derivadas de la izquierda
            if k % (N - 2) == 0 and k != 0 and k != (N - 2) * (M - 3):
                J[k, k] = 1/8 - 1/8 * VX[m, 2] - constante
                casos[k, k] = 5

                J[k, k+1] = 1/4 - 1/8 * VX[m, 1]
                casos[k, k+1] = 1

                J[k, k+(N-2)] = 1/4 - 1/8 * vy
                casos[k, k+(N-2)] = 3
                
                J[k, k-(N-2)] = 1/4 + 1/8 * vy
                casos[k, k-(N-2)] = 4

                print((k, k), (k, k+1), (k, k+(N-2)), (k, k-(N-2)), (2, m), (1, m))
            
            #derivadas de la derecha
            for w in range(2, M - 2):
                if (k / (w * (N - 2) - 1)) == 1 and k != 0:
                    J[k, k] = 1/8 * VX[m, N-3] - constante
                    casos[k, k] = 5

                    J[k, k-1] = 1/4 + 1/8 * VX[m, N-2]
                    casos[k, k-1] = 2

                    J[k, k+(N-2)] = 1/4 - 1/8 * vy
                    casos[k, k+(N-2)] = 3
                
                    J[k, k-(N-2)] = 1/4 + 1/8 * vy
                    casos[k, k-(N-2)] = 4

                    print((k, k), (k, k-1), (k, k+(N-2)), (k, k-(N-2)), (m, N-3), (m, N-2))
            
            #derivadas de arriba
            if 1 <= k <= (N - 4):
                J[k, k] = 1/8 * VX[m, n-1] - 1/8 * VX[m, n+1] - constante
                casos[k, k] = 5

                J[k, k+1] = 1/4 - 1/8 * VX[1, n]
                casos[k, k+1] = 1

                J[k, k-1] = 1/4 + 1/8 * VX[1, n]
                casos[k, k-1] = 2

                J[k, k+(N-2)] = 1/4 - 1/8 * vy
                casos[k, k+(N-2)] = 3

                print((k, k), (k, k+1), (k, k-1), (k, k+(N-2)), (n-1, m), (n+1, m), (n, 1))

            #derivadas de abajo
            elif (((N - 2) * (M - 3)) + 1) <= k <= (((N - 2) * (M - 2)) - 2):
                J[k, k] = 1/8 * VX[m, n-1] - 1/8 * VX[m, n+1] - constante
                casos[k, k] = 5

                J[k, k+1] = 1/4 - 1/8 * VX[m, n]
                casos[k, k+1] = 1

                J[k, k-1] = 1/4 + 1/8 * VX[m, n]
                casos[k, k-1] = 2

                J[k, k-(N-2)] = 1/4 + 1/8 * vy
                casos[k, k-(N-2)] = 4

                print((k, k), (k, k+1), (k, k-1), (k, k-(N-2)), (n-1, m), (n+1, m), (n, m))
            
            #derivada cuadro esquina superior izquierda
            elif k == 0:
                J[k, k] = 1/8 - 1/8 * VX[1, 2] - constante
                casos[k, k] = 5

                J[k, k+1] = 1/4 - 1/8 * VX[1, 1]
                casos[k, k+1] = 1

                J[k, k+(N-2)] = 1/4 - 1/8 * vy
                casos[k, k+(N-2)] = 3

                print((k, k), (k, k+1), (k, k+(N-2)), (2, 1), (1, 1))

            #derivada cuadro esquina superior derecha
            elif k == (N - 3):
                J[k, k] = 1/8 * VX[1, N - 3] - constante
                casos[k, k] = 5

                J[k, k-1] = 1/4 + 1/8 * VX[1, N-2]
                casos[k, k-1] = 2

                J[k, k+(N-2)] = 1/4 - 1/8 * vy
                casos[k, k+(N-2)] = 3

                print((k, k), (k, k-1), (k, k+(N-2)), (N-3, 1) , (N-2, 1))
            
            #derivada cuadro esquina inferior izquierda
            elif k == ((N - 2) * (M - 3)):
                J[k, k] = 1/8 - 1/8 * VX[M-2, 2] - constante
                casos[k, k] = 5

                J[k, k+1] = 1/4 - 1/8 * VX[M-2, 1]
                casos[k, k+1] = 1

                J[k, k-(N-2)] = 1/4 + 1/8 * vy
                casos[k, k-(N-2)] = 4

                print((k, k), (k, k+1), (k, k-(N-2)), (M-2, 2), (M-2, 1))

            #derivada cuadro esquina inferior derecha
            elif k == ((N - 2) * (M - 2) - 1):
                J[k, k] = 1/8 * VX[M-2, N-3] - constante
                casos[k, k] = 5

                J[k, k-1] = 1/4 + 1/8 * VX[M-2, N-2]
                casos[k, k-1] = 2

                J[k, k-(N-2)] = 1/4 + 1/8 * vy
                casos[k, k-(N-2)] = 4

                print((k, k), (k, k-1), (k, k-(N-2)), (M-2, N-3), (M-2, N-2))

            else:
                0

    return J.tocsr(), casos

# Construir Jacobiano y obtener la matriz de casos
J, casos = construir_J(vel_x, 0, N, M)
print(J.shape)

# # Graficar la matriz Jacobiana con colores según el caso de derivada
plt.figure(figsize=(8, 8))
plt.spy(J, markersize=2, color="black")  # Mostrar la estructura de la matriz J
plt.imshow(casos, cmap="coolwarm", alpha=0.5)  # Superponer la matriz de casos con colores

# Leyenda de colores
plt.colorbar(label="Caso de derivada")
plt.title("Estructura de la matriz Jacobiana con casos diferenciados")
plt.xlabel("Índice de columna")
plt.ylabel("Índice de fila")
plt.show()

cond_J = spla.norm(J, ord=2) * spla.norm(spla.inv(J), ord=2)

if np.isinf(cond_J) or np.isnan(cond_J):
    print("La matriz J es singular o mal condicionada.")
else:
    print(f"Condición de J: {cond_J}")

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

vel_x_final = resolver_sistema(vel_x, 0, N, M, 1, 5, 5)
graficar_resultado(vel_x_final)