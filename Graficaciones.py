import matplotlib.pyplot as plt

def Graficar_Matriz(matriz, titulo, origen, vmin=None, vmax=None):
    plt.figure(figsize=(10, 6))
    plt.title(titulo)
    
    # Cambiar el colormap a uno pastel y simétrico
    im = plt.imshow(matriz, cmap='coolwarm', origin=origen, vmin=vmin, vmax=vmax)
    
    plt.colorbar(im, label='Valor')
    plt.xlabel('Columna')
    plt.ylabel('Fila')
    plt.grid(False)
    plt.show()
