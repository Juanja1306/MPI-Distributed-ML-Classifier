from mpi4py import MPI
import socket
import psutil
import platform

# Configuración inicial MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)

# --- NODO MÁSTER (Rank 0) ---
if rank == 0:
    print("="*50)
    print("      DIAGNÓSTICO DE CLUSTER MPI - UPS")
    print("="*50)
    print(f"[MASTER] Nodo Principal detectado.")
    print(f"[MASTER] Hostname: {hostname} | IP: {ip_address}")
    print(f"[MASTER] Esperando respuesta de {size - 1} esclavo(s)...\n")

    # OPTIMIZACIÓN: Usar ANY_SOURCE
    # Esto previene que el programa se cuelgue esperando al nodo 1 si el nodo 2 ya está listo.
    for _ in range(1, size):
        status = MPI.Status()
        # Recibimos de CUALQUIER nodo que responda primero
        data = comm.recv(source=MPI.ANY_SOURCE, status=status)
        source_rank = status.Get_source() # Identificamos quién envió el mensaje
        
        print(f"--- NODO ESCLAVO {source_rank} DETECTADO ---")
        print(f"  > Hostname: {data['host']}")
        print(f"  > IP Local: {data['ip']}")
        print(f"  > Sistema:  {data['os']}")
        print(f"  > CPU Cores:{data['cores']}")
        print(f"  > RAM Total:{data['ram']} GB")
        print("-" * 35)

    print("\n[RESULTADO] Todos los nodos responden correctamente.")
    print("El clúster está listo para el entrenamiento distribuido.")
    print("="*50)

# --- NODOS ESCLAVOS (Rank > 0) ---
else:
    # Recopilar información técnica del nodo esclavo
    try:
        info = {
            'host': hostname,
            'ip': ip_address,
            'os': platform.system() + " " + platform.release(),
            'cores': psutil.cpu_count(logical=False),
            'ram': round(psutil.virtual_memory().total / (1024**3), 2)
        }
        # Enviar al Máster
        comm.send(info, dest=0)
    except Exception as e:
        # En caso de error en el esclavo, intentamos enviar el error al master
        error_info = {
            'host': hostname,
            'ip': ip_address,
            'os': f"ERROR: {str(e)}",
            'cores': 0,
            'ram': 0
        }
        comm.send(error_info, dest=0)