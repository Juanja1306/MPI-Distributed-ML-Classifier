from mpi4py import MPI
import sys

def prueba():
    try:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        name = MPI.Get_processor_name()
        
        print(f"Hola! Soy el proceso {rank} de {size} ejecutandome en {name}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    prueba()