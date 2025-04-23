from Construcciones import *
import numpy as np

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

# Método de Jacobi para resolver el sistema de ecuaciones no lineales
def Jacobi(J, F, max_iter=100, tol=1e-4):
    n = len(F)
    H = np.zeros_like(F)
    for it in range(max_iter):
        H_nuevo = np.zeros_like(H)
        for i in range(n):
            suma = np.dot(J[i, :], H) - J[i, i] * H[i]
            H_nuevo[i] = (-F[i] - suma) / J[i, i]
        if np.linalg.norm(H_nuevo - H, np.inf) < tol:
            return H_nuevo
        H = H_nuevo.copy()
    return H

# Método de Gradiente Descendente para resolver el sistema de ecuaciones no lineales
def Gradiente_Descendente(J, H, F, N, M, ITER):
    VX = H
    H = H[1:M-1, 1:N-1].flatten()

    if (np.allclose(J, J.T, atol = 1e-8) and np.dot(np.dot (H.T, J), H) >= 0):
        for i in range (1, ITER+1):
            G = -1 * (F + np.dot(J, H))
            alpha = np.dot(G.T, G) / (np.dot(np.dot(G.T, J), G))
            H = H + alpha * G
            print (f"Iteración {i}")

        VX[1:M-1, 1:N-1] = VX[1:M-1, 1:N-1] + H.reshape((M-2, N-2))

        return VX
    else:
        return 0

# Método de Gradiente Conjugado para resolver el sistema de ecuaciones no lineales
def Gradiente_Conjugado(J, H, F, N, M, ITER, tol1, tol2):
    VX = H
    H = H[1:M-1, 1:N-1].flatten()

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

        VX[1:M-1, 1:N-1] = VX[1:M-1, 1:N-1] + H.reshape((M-2, N-2))

        return VX
    else:
        return 0