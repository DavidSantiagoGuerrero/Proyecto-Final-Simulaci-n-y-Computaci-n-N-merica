import matplotlib.pyplot as plt

def Graficar_Matriz(matriz, titulo, origen):
    plt.figure(figsize=(12, 6))
    plt.imshow(matriz, cmap='jet', interpolation='nearest', origin=origen, aspect='auto')
    plt.colorbar(label='v_x')
    plt.title(titulo, fontsize=14)
    plt.xlabel('Índice i', fontsize=12)
    plt.ylabel('Índice j', fontsize=12)
    plt.show()
