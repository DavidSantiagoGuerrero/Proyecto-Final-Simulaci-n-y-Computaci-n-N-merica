from Construcciones import *
import numpy as np
from numpy import linalg

# Método de Newton-Raphson para resolver el sistema de ecuaciones no lineales
def Newton_Raphson(VX, vy, N, M, relajacion, tolerancia, precision, ITER, ITER_OG):
    print(f"Iteración {ITER_OG - ITER + 1}")

    F = construir_F(VX, vy, N, M, precision)
    J = construir_J(VX, vy, N, M, precision)
    H = np.linalg.solve(J, -F)
    H = relajacion * H

    Norma = np.min(VX[1:M-1, 1:N-1])

    if Norma <= tolerancia or ITER == 1:
        return VX
        
    nuevo_vx = VX.copy()
    nuevo_vx[1:M-1, 1:N-1] = VX[1:M-1, 1:N-1] + np.reshape(H, (M-2, N-2))

    return Newton_Raphson(nuevo_vx, vy, N, M, relajacion, tolerancia, precision, ITER-1, ITER_OG)

# Verificar convergencia de Richardson
def verificar_convergencia_richardson(VX, vy, N, M, precision):
    J = construir_J(VX, vy, N, M, precision)
    I = np.eye(J.shape[0])
    B = I - J
    norma_B = np.linalg.norm(B, ord=np.inf)
    print(f"Norma ||I - J||_∞ = {norma_B}")

    return norma_B < 1

# Método de Jacobi para resolver el sistema de ecuaciones no lineales
# def Jacobi(J, F, max_iter=100, tol=1e-4):
#     n = len(F)
#     H = np.zeros_like(F)
#     for it in range(max_iter):
#         H_nuevo = np.zeros_like(H)
#         for i in range(n):
#             suma = np.dot(J[i, :], H) - J[i, i] * H[i]
#             H_nuevo[i] = (-F[i] - suma) / J[i, i]
#         if np.linalg.norm(H_nuevo - H, np.inf) < tol:
#             return H_nuevo
#         H = H_nuevo.copy()
#     return H

# Método de Jacobi para resolver el sistema de ecuaciones no lineales
def Jacobi(VX, vy, N, M, relajacion, tolerancia, precision, ITER, ITER_OG, max_iter_jacobi=100):
    print(f"Iteración {ITER_OG - ITER + 1}")

    # 1. Calcular F y J
    F = construir_F(VX, vy, N, M, precision)
    J = construir_J(VX, vy, N, M, precision)

    # 2. Resolver J·H = -F con Jacobi interno
    n = len(F)
    H = np.zeros_like(F)

    for _ in range(max_iter_jacobi):
        H_nuevo = np.zeros_like(H)
        for i in range(n):
            suma = 0
            for j in range(n):
                if j != i:
                    suma += J[i, j] * H[j]
            H_nuevo[i] = (-F[i] - suma) / J[i, i]

        # Criterio de parada del Jacobi interno
        if np.linalg.norm(H_nuevo - H, np.inf) < tolerancia:
            break

        H = H_nuevo.copy()

    # 3. Aplicar relajación
    H *= relajacion

    # 4. Verificar convergencia global
    if np.linalg.norm(H, np.inf) < tolerancia or ITER == 1:
        return VX

    # 5. Actualizar la solución VX
    nuevo_vx = VX.copy()
    nuevo_vx[1:M-1, 1:N-1] += np.reshape(H, (M-2, N-2))

    # 6. Llamada recursiva
    return Jacobi(nuevo_vx, vy, N, M, relajacion, tolerancia, precision, ITER - 1, ITER_OG, max_iter_jacobi)

# def gauss_seidel(J, F, max_iter=100, tol=1e-4):
#     n = len(F)
#     H = np.zeros_like(F)

#     for it in range(max_iter):
#         H_anterior = H.copy()

#         for i in range(n):
#             suma = 0
#             for j in range(n):
#                 if j != i:
#                     suma += J[i, j] * H[j]  # H[j] puede ser el nuevo valor ya actualizado
#             H[i] = (-F[i] - suma) / J[i, i]

#         if np.linalg.norm(H - H_anterior, np.inf) < tol:
#             return H

#     return H

# Método de Gauss-Seidel para resolver el sistema de ecuaciones no lineales
def Gauss_Seidel(VX, vy, N, M, relajacion, tolerancia, precision, ITER, ITER_OG, max_iter_gs=100):
    print(f"Iteración {ITER_OG - ITER + 1}")

    # 1. Calcular F y J
    F = construir_F(VX, vy, N, M, precision)
    J = construir_J(VX, vy, N, M, precision)

    # 2. Resolver J·H = -F con Gauss-Seidel interno
    n = len(F)
    H = np.zeros_like(F)

    for _ in range(max_iter_gs):
        H_anterior = H.copy()
        for i in range(n):
            suma = 0
            for j in range(n):
                if j != i:
                    suma += J[i, j] * H[j]  # Usa valores nuevos si ya se actualizaron
            H[i] = (-F[i] - suma) / J[i, i]

        # Criterio de parada del Gauss-Seidel interno
        if np.linalg.norm(H - H_anterior, np.inf) < tolerancia:
            break

    # 3. Aplicar relajación
    H *= relajacion

    # 4. Verificar convergencia global
    if np.linalg.norm(H, np.inf) < tolerancia or ITER == 1:
        return VX

    # 5. Actualizar la solución VX
    nuevo_vx = VX.copy()
    nuevo_vx[1:M-1, 1:N-1] += np.reshape(H, (M-2, N-2))

    # 6. Llamada recursiva
    return Gauss_Seidel(nuevo_vx, vy, N, M, relajacion, tolerancia, precision, ITER - 1, ITER_OG, max_iter_gs)

# Método de Gradiente Descendente para resolver el sistema de ecuaciones no lineales
def Gradiente_Descendente(VX, vy, N, M, ITER, precision, tol):
    F = construir_F(VX, vy, N, M, precision).copy()
    J = construir_J(VX, vy, N, M, precision).copy()
    Vx = VX.copy()
    H = Vx[1:M-1, 1:N-1].flatten()

    print("Condicion: ", np.linalg.cond(J, np.inf))

    if (np.allclose(J, J.T, atol = 1e-8) and np.dot(np.dot (H.T, J), H) >= 0):
        for i in range (1, ITER+1):
            print (f"Iteración {i}")
            G = -1 * (F + np.dot(J, H))
            alpha = np.dot(G.T, G) / (np.dot(np.dot(G.T, J), G))
            H = H + alpha * G

        Vx[1:M-1, 1:N-1] = H.reshape((M-2, N-2))

        return Vx
    else:
        return 0

# Método de Gradiente Conjugado para resolver el sistema de ecuaciones no lineales
def Gradiente_Conjugado(VX, vy, N, M, ITER, precision, tol1, tol2):
    F = construir_F(VX, vy, N, M, precision).copy()
    J = construir_J(VX, vy, N, M, precision).copy()
    Vx = VX.copy()
    H = Vx[1:M-1, 1:N-1].flatten()

    print("Condicion: ", np.linalg.cond(J, np.inf))

    if (np.allclose(J, J.T, atol = 1e-8) and np.dot(np.dot (H.T, J), H) >= 0):
        G = -1 * (F.T - np.dot(J, H))
        v = G
        c = np.dot(G, G.T)

        for i in range (1, ITER+1):
            if ((np.dot(v.T, v))**(1/2) < tol1):
                break
            alpha = c / (np.dot(np.dot(v.T, J), v))
            H = H + alpha * v
            G = G - alpha * np.dot(J, v)
            d = np.dot(G.T, G)
            if (d < tol2):
                break
            v = G + (d / c) * v
            c = d
            print (f"Iteración {i}")

        Vx[1:M-1, 1:N-1] = H.reshape((M-2, N-2))
        
        return Vx
    else:
        return 0