import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def Graficar_Matriz(matriz, titulo, origen):
    plt.figure(figsize=(10, 6))
    plt.title(titulo)
    
    im = plt.imshow(matriz, cmap = 'coolwarm', origin = origen)
    cbar = plt.colorbar(im, format="%.10f")
    cbar.set_label('Valor')

    plt.xlabel('Columna')
    plt.ylabel('Fila')
    plt.grid(False)
    plt.show()