from mpi4py import MPI
import pandas as pd
import time
import numpy as np

# Librerías de Machine Learning (NLP)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Librerías para GUI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# Configuración Inicial MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ==========================================
# LÓGICA DEL NODO MÁSTER (RANK 0)
# ==========================================
if rank == 0:
    print(f"[Master] Iniciando en nodo {rank}. Esperando {size-1} trabajadores.")
    
    # 1. Cargar Dataset (Simulado para que funcione directo, puedes cambiar por pd.read_csv)
    # Requisito: Columna de texto y etiquetas [cite: 27, 28, 29]
    data = {
        'texto': [
            "oferta increible gana dinero rapido", "hola como estas amigo", 
            "compra ahora y gana premio", "reunion de trabajo mañana", 
            "tienes un nuevo mensaje de voz", "ganaste la loteria",
            "informe del proyecto adjunto", "gratis cupones de descuento",
            "vamos a almorzar hoy", "urgente respuesta requerida banco",
            "te amo mucho mi vida", "factura pendiente de pago",
            "clase de computacion paralela", "viagra barato envio rapido",
            "partido de futbol el domingo", "casino online bono gratis",
            "receta de cocina facil", "click aqui para premios",
            "documentos legales del juicio", "pierde peso rapidamente pastillas"
        ],
        'etiqueta': [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1] 
        # 1 = Spam, 0 = Ham
    }
    df = pd.DataFrame(data)
    
    print("[Master] Vectorizando datos de texto (TF-IDF)...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['texto'])
    y = df['etiqueta']

    # Dividir datos (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Enviar datos a los nodos hijos [cite: 39]
    # Enviaremos todo el paquete necesario para entrenar
    paquete_datos = {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test,   'y_test': y_test
    }

    # Enviar al Nodo 1 (Modelo Naive Bayes)
    print("[Master] Enviando datos al Nodo 1...")
    comm.send({'tipo_modelo': 'NaiveBayes', 'datos': paquete_datos}, dest=1)

    # Enviar al Nodo 2 (Modelo Regresión Logística)
    print("[Master] Enviando datos al Nodo 2...")
    comm.send({'tipo_modelo': 'LogisticRegression', 'datos': paquete_datos}, dest=2)

    # 3. Recibir resultados
    print("[Master] Esperando resultados de los nodos hijos...")
    resultados = []
    
    # Recibimos de los ranks 1 y 2
    for i in range(1, 3):
        res = comm.recv(source=i)
        resultados.append(res)
        print(f"[Master] Resultados recibidos del Nodo {i}: {res['modelo']}")

    # 4. GUI - Interfaz Gráfica 
    # La GUI solo se ejecuta en el master [cite: 53]
    def mostrar_gui():
        root = tk.Tk()
        root.title("Resultados Entrenamiento Distribuido MPI")
        root.geometry("800x600")

        lbl_titulo = tk.Label(root, text="Panel de Control - Nodo Máster", font=("Arial", 16, "bold"))
        lbl_titulo.pack(pady=10)

        # Tabla de métricas
        cols = ("Nodo", "Modelo", "Accuracy", "Tiempo (s)")
        tree = ttk.Treeview(root, columns=cols, show='headings')
        for col in cols:
            tree.heading(col, text=col)
        
        nombres_modelos = []
        accuracies = []
        tiempos = []

        for r in resultados:
            tree.insert("", "end", values=(r['rank'], r['modelo'], f"{r['accuracy']:.4f}", f"{r['tiempo']:.4f}"))
            nombres_modelos.append(r['modelo'])
            accuracies.append(r['accuracy'])
            tiempos.append(r['tiempo'])

        tree.pack(pady=20, padx=20, fill='x')

        # Gráfico Comparativo
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        
        # Gráfico de Precisión
        ax[0].bar(nombres_modelos, accuracies, color=['blue', 'green'])
        ax[0].set_title('Comparación de Accuracy')
        ax[0].set_ylim(0, 1.1)
        
        # Gráfico de Tiempo
        ax[1].bar(nombres_modelos, tiempos, color=['red', 'orange'])
        ax[1].set_title('Tiempo de Entrenamiento (s)')
        
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack()

        root.mainloop()

    print("[Master] Lanzando GUI...")
    mostrar_gui()

# ==========================================
# LÓGICA DE LOS NODOS HIJOS (RANK 1 y 2)
# ==========================================
else:
    # 1. Recibir datos del Master
    paquete = comm.recv(source=0)
    datos = paquete['datos']
    tipo = paquete['tipo_modelo']
    
    X_train = datos['X_train']
    y_train = datos['y_train']
    X_test = datos['X_test']
    y_test = datos['y_test']

    print(f"[Nodo {rank}] Datos recibidos. Entrenando modelo: {tipo}")

    # 2. Entrenar Modelo 
    modelo = None
    start_time = time.time()

    if rank == 1:
        # Nodo 1 entrena Naive Bayes
        modelo = MultinomialNB()
    elif rank == 2:
        # Nodo 2 entrena Regresión Logística (n_jobs=-1 para paralelismo interno )
        modelo = LogisticRegression(max_iter=1000, n_jobs=-1)
    
    # Entrenamiento
    if modelo:
        modelo.fit(X_train, y_train)
        
        # Predicción y Métricas
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        end_time = time.time()
        tiempo_total = end_time - start_time

        print(f"[Nodo {rank}] Entrenamiento finalizado. Accuracy: {acc:.2f}")

        # 3. Enviar resultados al Master
        resultado_paquete = {
            'rank': rank,
            'modelo': tipo,
            'accuracy': acc,
            'tiempo': tiempo_total
        }
        comm.send(resultado_paquete, dest=0)