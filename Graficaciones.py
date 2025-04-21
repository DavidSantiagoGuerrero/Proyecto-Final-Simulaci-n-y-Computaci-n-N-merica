import matplotlib.pyplot as plt

def Graficar_Matriz(vx_final):
    plt.figure(figsize=(12, 6))
    plt.imshow(vx_final, cmap='jet', interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(label='v_x')
    plt.title('Distribución final de v_x', fontsize=14)
    plt.xlabel('Índice i', fontsize=12)
    plt.ylabel('Índice j', fontsize=12)
    plt.show()

def Graficar_Jacobiano(J):
    plt.figure(figsize=(12, 6))
    plt.imshow(J, cmap='jet', interpolation='nearest', origin='upper', aspect='auto')
    plt.colorbar(label='J')
    plt.title('Gráfica de J(X)', fontsize=14)
    plt.xlabel('Índice i', fontsize=12)
    plt.ylabel('Índice j', fontsize=12)
    plt.show()