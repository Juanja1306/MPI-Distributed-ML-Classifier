# Distributed NLP Training with MPI

Este proyecto implementa un sistema de clasificación de texto (Spam vs Ham) utilizando computación paralela.

## 🚀 Tecnologías
- **Python 3.12**
- **MPI4Py** (Message Passing Interface)
- **Scikit-Learn** (Modelos ML)
- **Tkinter** (Visualización de resultados en el Nodo Maestro)

## ⚙️ Arquitectura
El sistema funciona con un esquema **Master-Slave**:
1. **Nodo 0 (Master):** Preprocesa el texto (TF-IDF), distribuye los datos y visualiza métricas.
2. **Nodo 1 (Worker):** Entrena un modelo Naive Bayes.
3. **Nodo 2 (Worker):** Entrena un modelo de Regresión Logística (con paralelismo interno).

## 📦 Ejecución
```bash
mpiexec -n 3 -f hostfile python main_mpi.py


---
mpiexec -hosts 2 192.168.0.151 1 192.168.0.100 1 hostname


Master -> 192.168.0.151
Esclava -> 192.168.0.100

---
Escuchar en TODAS
smpd -d
