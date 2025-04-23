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
vy = 1/8

# Definir el número máximo de iteraciones, tolerancia, cantidad de decimales y factor de relajación
i = 300
tolerancia = 3e-3
decimales = 20
factor_relajacion = 0.1

# Prueba a cada método
print("--------------------PRUEBA DE NEWTON-RAPHSON--------------------")
VX_NEWTON = Newton_Raphson(vel_x_2, vy, Columnas, Filas, factor_relajacion, tolerancia, decimales, i, i)
print("--------------------PRUEBA DE RICHARDSON--------------------")
VX_RICHARDSON = verificar_convergencia_richardson(vel_x_2, vy, Columnas, Filas, decimales)
print("--------------------PRUEBA DE JACOBI------------------")
VX_JACOBI = Jacobi(vel_x_2, vy, Columnas, Filas, factor_relajacion, tolerancia, decimales, i, i)
print("--------------------PRUEBA DE GAUSS-SEIDEL------------------")
VX_GAUSS = Gauss_Seidel(vel_x_2, vy, Columnas, Filas, factor_relajacion, tolerancia, decimales, i, i)
print("--------------------PRUEBA DE GRADIENTE DESCENDENTE------------------")
# VX_GD = Gradiente_Descendente(vel_x_4, vy, Columnas, Filas, i, decimales, tolerancia)
print("----------------------PRUEBA DE GRADIENTE CONJUGADO--------------------")
VX_GC = Gradiente_Conjugado(vel_x_4, 0, Columnas, Filas, i, decimales, 1e-5, 1e-5)

# Graficar resultados de cada método
Graficar_Matriz(VX_NEWTON, "Newton Raphson", 'lower')

Graficar_Matriz(VX_JACOBI, "Jacobi", 'lower')

Graficar_Matriz(VX_GAUSS, "Gauss-Seidel", 'lower')

# if type(VX_GD) == int:
#     print("No se puede aplicar el método del gradiente descendente\nporque la matriz no es simétrica o no es positiva definida")
# else:
#     Graficar_Matriz(VX_GD, "Gradiente Descendente", 'lower')

if type(VX_GC) == int:
    print("No se puede aplicar el método del gradiente conjugado\nporque la matriz no es simétrica o no es positiva definida")
else:
    Graficar_Matriz(VX_GC, "Gradiente Conjugado", 'lower')