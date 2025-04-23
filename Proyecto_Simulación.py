# Importar las funciones de Inicializaciones, Graficaciones, Construcciones y Metodos_Resolver_Sistemas
from Inicializaciones import *
from Graficaciones import *
from Construcciones import *
from Metodos_Resolver_Sistemas import *

# Definir dimensiones de la malla
Columnas, Filas = 50, 8

# Inicializar matrizes de velocidad
vel_x_1 = Matriz_Redución_Completa(Filas, Columnas, 10)
vel_x_2 = Matriz_Redución_Eje_X(Filas, Columnas, 10)
vel_x_3 = Matriz_Velocidades_Uniforme(Filas, Columnas, 1)
vel_x_4 = Matriz_Velocidades_Uniforme(Filas, Columnas, 0)

# Definir el número máximo de iteraciones, tolerancia, cantidad de decimales y factor de relajación
i = 100
tolerancia = 3e-3
decimales = 20
factor_relajacion = 0.1

F_GD = construir_F(vel_x_4, 0, Columnas, Filas, decimales)
F_GC = construir_F(vel_x_4, 0, Columnas, Filas, decimales)

J_GD = construir_J(vel_x_4, 0, Columnas, Filas, decimales)
J_GC = construir_J(vel_x_4, 0, Columnas, Filas, decimales)

H_GD = vel_x_4
H_GC = vel_x_4

# Prueba a cada método
print("--------------------PRUEBA DE NEWTON-RAPHSON--------------------")
VX_NEWTON= Newton_Raphson(vel_x_2, 0, Columnas, Filas, factor_relajacion, tolerancia, decimales, i, i)
print("----------------------PRUEBA DE GRADIENTE DESCENDENTE--------------------")
VX_GD = Gradiente_Descendente(J_GD, H_GD, F_GD, Columnas, Filas, i)
# print("----------------------PRUEBA DE GRADIENTE CONJUGADO--------------------")
# VX_GC = Gradiente_Conjugado(J_GC, H_GC, F_GC, Columnas, Filas, i, 1e-3, 1e-3)

# Graficar resultados de cada método
Graficar_Matriz(VX_NEWTON, "Newton Raphson", 'lower')

if type(VX_GD) == int:
    print("No se puede aplicar el método del gradiente descendente\nporque la matriz no es simétrica o no es positiva definida")
else:
    Graficar_Matriz(VX_GD, "Gradiente Descendente", 'lower')

# if type(VX_GC) == int:
#     print("No se puede aplicar el método del gradiente conjugado\nporque la matriz no es simétrica o no es positiva definida")
# else:
#     Graficar_Matriz(VX_GC, "Gradiente Conjugado", 'lower')