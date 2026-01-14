from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

print(f"Hola! Soy el proceso {rank} de {size} ejecutandome en {name}")
