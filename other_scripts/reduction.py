import numpy
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

array_size = size
# Se cambió numpy.int por int o numpy.int64
recvdata = numpy.zeros(array_size, dtype=int)
senddata = (rank+1)*numpy.arange(array_size, dtype=int)

print(" process %s sending %s " %(rank , senddata))

# Operación de reducción: suma todos los arreglos en el proceso raíz (root 0)
comm.Reduce(senddata, recvdata, root=0, op=MPI.SUM)

if rank == 0:
    print('on task', rank, 'after Reduce:    data = ', recvdata)
else:
    print('on task', rank, 'finished reduction step')