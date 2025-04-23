import numpy as np

# Función para calcular el vector F(X) evaluado en V
def construir_F(VX, vy, N, M, precision):
    # Inicializar la matriz F
    F = np.zeros((M-2, N-2), dtype=float)
    
    for m in range (1, M-1):
        for n in range (1, N-1):
            # Calcular la ecuación para el punto (m, n)
            eq = round((1/4) * (VX[m, n+1] + VX[m, n-1] + VX[m+1, n] + VX[m-1, n])
                       + (1/8) * (VX[m, n] * (VX[m, n-1] - VX[m, n+1])
                       + vy * (VX[m-1, n] - VX[m+1, n])) - VX[m, n], precision)
            
            # Asignar el valor de la ecuación a la matriz F
            F[m-1, n-1] = eq

    return F.flatten()

# Función para calcular la matrix J(X) evaluada en V
def construir_J(VX, vy, N, M, precision):
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
                #J[k, k] = round(1/8 * (1 - VX[m, 2]) - 1, precision)
                J[k, k] = -1
                J[k, k+1] = round(1/4 - 1/8 * VX[m, 1], precision)
                J[k, k+(N-2)] = round(1/4 - 1/8 * vy, precision)
                J[k, k-(N-2)] = round(1/4 + 1/8 * vy, precision)
                
            #derivadas de la derecha
            elif k in bordes_der:
                #J[k, k] = round(1/8 * VX[m, N-3] - 1, precision)
                J[k, k] = -1
                J[k, k-1] = round(1/4 + 1/8 * VX[m, N-2], precision)
                J[k, k+(N-2)] = round(1/4 - 1/8 * vy, precision)
                J[k, k-(N-2)] = round(1/4 + 1/8 * vy, precision)

            #derivadas del centro
            elif N - 1 <= k and k <= ((N - 2) * (M - 3)) - 2 and k not in bordes_izq and k not in bordes_der:
                #J[k, k] = round(1/8 * (VX[m, n-1] - VX[m, n+1]) - 1, precision)
                J[k, k] = -1
                J[k, k+1] = round(1/4 - 1/8 * VX[m, n], precision)
                J[k, k-1] = round(1/4 + 1/8 * VX[m, n], precision)
                J[k, k+(N-2)] = round(1/4 - 1/8 * vy, precision)
                J[k, k-(N-2)] = round(1/4 + 1/8 * vy, precision)
                    
            #derivadas de arriba
            elif 1 <= k <= (N - 4):
                #J[k, k] = round(1/8 * (VX[m, n-1] - VX[m, n+1]) - 1, precision)
                J[k, k] = -1
                J[k, k+1] = round(1/4 - 1/8 * VX[1, n], precision)
                J[k, k-1] = round(1/4 + 1/8 * VX[1, n], precision)
                J[k, k+(N-2)] = round(1/4 - 1/8 * vy, precision)
                
            #derivadas de abajo
            elif (((N - 2) * (M - 3)) + 1) <= k <= (((N - 2) * (M - 2)) - 2):
                #J[k, k] = round(1/8 * (VX[m, n-1] - VX[m, n+1]) - 1, precision)
                J[k, k] = -1
                J[k, k+1] = round(1/4 - 1/8 * VX[m, n], precision)
                J[k, k-1] = round(1/4 + 1/8 * VX[m, n], precision)
                J[k, k-(N-2)] = round(1/4 + 1/8 * vy, precision)
                
            #derivada cuadro esquina superior izquierda
            elif k == 0:
                #J[k, k] = round(1/8 * (1 - VX[1, 2]) - 1, precision)
                J[k, k] = -1
                J[k, k+1] = round(1/4 - 1/8 * VX[1, 1], precision)
                J[k, k+(N-2)] = round(1/4 - 1/8 * vy, precision)
                
            #derivada cuadro esquina superior derecha
            elif k == (N - 3):
                #J[k, k] = round(1/8 * VX[1, N - 3] - 1, precision)
                J[k, k] = -1
                J[k, k-1] = round(1/4 + 1/8 * VX[1, N-2], precision)
                J[k, k+(N-2)] = round(1/4 - 1/8 * vy, precision)
                
            #derivada cuadro esquina inferior izquierda
            elif k == ((N - 2) * (M - 3)):
                #J[k, k] = round(1/8 * (1 - VX[M-2, 2]) - 1, precision)
                J[k, k] = -1
                J[k, k+1] = round(1/4 - 1/8 * VX[M-2, 1], precision)
                J[k, k-(N-2)] = round(1/4 + 1/8 * vy, precision)

            #derivada cuadro esquina inferior derecha
            elif k == ((N - 2) * (M - 2) - 1):
                #J[k, k] = round(1/8 * VX[M-2, N-3] - 1, precision)
                J[k, k] = -1
                J[k, k-1] = round(1/4 + 1/8 * VX[M-2, N-2], precision)
                J[k, k-(N-2)] = round(1/4 + 1/8 * vy, precision)
                
            else:
                0
    return J
