import numpy as np
from Proyecto_Simulación import *

def gradiente_descendente(J, H, F, N, M, ITER):
    H = H[1:M-1, 1:N-1].flatten()
    if (np.allclose(J, J.T, atol = 1e-8) and np.dot(np.dot (H.T, J), H) >= 0):
        for i in range (1, ITER+1):
            G = F.T - np.dot(J, H)
            alpha = np.dot(G.T, G) / (np.dot(np.dot(G.T, J), G))
            H = H - alpha * G
        
        return i, H
    else:
        print("No se puede aplicar el método del gradiente descendente\nporque la matriz no es simétrica o no es positiva definida")

def gradiente_conjugado(J, H, F, N, M, ITER, tol1, tol2):
    H = H[1:M-1, 1:N-1].flatten()
    if (np.allclose(J, J.T, atol = 1e-8) and np.dot(np.dot (H.T, J), H) >= 0):
        G = F.T - np.dot(J, H)
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

        return i, H
    else:
        print("No se puede aplicar el método del gradiente conjugado\nporque la matriz no es simétrica o no es positiva definida")