# mpiexec -n 10 python hello.py

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"Hello, World! from rank {rank}")
# Saca un nuemero aleatorio de 1 a 1000000 y saca su raiz cuadrada sin librerias externas
import random
numero = random.randint(1, 1000000)
raiz = numero ** 0.5
print(f"El numero {numero} tiene una raiz cuadrada de {raiz} del proceso {rank}")