import sys
import os
import time
import pandas as pd
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
from mpi4py import MPI
import pickle
import zlib

# --- IMPORTS PARA GRÁFICAS ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Librerías de Machine Learning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configurar encoding para evitar errores Unicode en Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# --- Configuración del entorno MPI ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

TAG_RESULT = 22

# ==========================================
# UTILIDADES DE COMPRESIÓN
# ==========================================
def compress_data(data):
    """Comprime datos usando pickle + zlib para transferencia rápida"""
    pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(pickled, level=6)
    return compressed

def decompress_data(compressed):
    """Descomprime datos"""
    pickled = zlib.decompress(compressed)
    data = pickle.loads(pickled)
    return data

# ==========================================
# VENTANA DE MÉTRICAS (POPUP)
# ==========================================
class MetricsWindow:
    def __init__(self, parent, results):
        self.window = tk.Toplevel(parent)
        self.window.title("Metricas Detalladas del Cluster")
        self.window.geometry("1400x900")
        self.window.configure(bg="#f0f0f0")
        
        # Frame principal con scroll
        main_frame = tk.Frame(self.window, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas con scrollbar
        canvas = tk.Canvas(main_frame, bg="white")
        scrollbar_y = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollbar_x = tk.Scrollbar(main_frame, orient="horizontal", command=canvas.xview)
        
        inner_frame = tk.Frame(canvas, bg="white")
        
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        inner_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Dibujar gráficas
        self.draw_charts(inner_frame, results)
        
        # Botón cerrar
        btn_close = tk.Button(self.window, text="Cerrar", command=self.window.destroy,
                            bg="#e74c3c", fg="white", font=("Arial", 11, "bold"), 
                            padx=20, pady=8)
        btn_close.pack(pady=10)
    
    def draw_charts(self, parent, results):
        """Visualización completa de métricas"""
        models = [res['model'] for res in results]
        accuracies = [res['accuracy'] * 100 for res in results]
        times = [res['time'] for res in results]
        f1_scores = [res['f1_score'] * 100 for res in results]
        precisions = [res['precision'] * 100 for res in results]
        recalls = [res['recall'] * 100 for res in results]
        colors = ['#27ae60', '#3498db', '#e74c3c'][:len(models)]

        # Crear figura grande
        fig = plt.figure(figsize=(16, 11), dpi=100)
        fig.patch.set_facecolor('white')
        
        # Grid de 3x2
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
        
        # === GRÁFICA 1: Accuracy ===
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(range(len(models)), accuracies, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        ax1.set_title('Accuracy por Modelo', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('Porcentaje (%)', fontsize=12)
        ax1.set_ylim(0, 105)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=20, ha='right', fontsize=11)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
            height = bar.get_height()
            ax1.text(i, height + 1.5, f'{acc:.2f}%', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # === GRÁFICA 2: F1-Score ===
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(range(len(models)), f1_scores, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        ax2.set_title('F1-Score por Modelo', fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylabel('Porcentaje (%)', fontsize=12)
        ax2.set_ylim(0, 105)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=20, ha='right', fontsize=11)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, (bar, f1) in enumerate(zip(bars2, f1_scores)):
            height = bar.get_height()
            ax2.text(i, height + 1.5, f'{f1:.2f}%', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # === GRÁFICA 3: Tiempo de Entrenamiento ===
        ax3 = fig.add_subplot(gs[1, 0])
        bars3 = ax3.barh(range(len(models)), times, color=colors, alpha=0.8, 
                        edgecolor='black', linewidth=1.5)
        ax3.set_title('Tiempo de Entrenamiento', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Segundos (s)', fontsize=12)
        ax3.set_yticks(range(len(models)))
        ax3.set_yticklabels(models, fontsize=11)
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        ax3.invert_yaxis()
        
        for i, time_val in enumerate(times):
            ax3.text(time_val + (max(times) * 0.02), i, f'{time_val:.3f}s', 
                    va='center', fontsize=11, fontweight='bold')

        # === GRÁFICA 4: Precision vs Recall ===
        ax4 = fig.add_subplot(gs[1, 1])
        x = range(len(models))
        width = 0.35
        bars4a = ax4.bar([i - width/2 for i in x], precisions, width, 
                         label='Precision', color='#3498db', alpha=0.8, 
                         edgecolor='black', linewidth=1)
        bars4b = ax4.bar([i + width/2 for i in x], recalls, width, 
                         label='Recall', color='#e74c3c', alpha=0.8, 
                         edgecolor='black', linewidth=1)
        ax4.set_title('Precision vs Recall', fontsize=14, fontweight='bold', pad=15)
        ax4.set_ylabel('Porcentaje (%)', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=20, ha='right', fontsize=11)
        ax4.set_ylim(0, 105)
        ax4.legend(loc='lower right', fontsize=11)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')

        # === GRÁFICA 5: Comparativa General ===
        ax5 = fig.add_subplot(gs[2, 0])
        width = 0.2
        x = range(len(models))
        
        bars5a = ax5.bar([i - width*1.5 for i in x], accuracies, width, 
                        label='Accuracy', color='#27ae60', alpha=0.8, edgecolor='black')
        bars5b = ax5.bar([i - width*0.5 for i in x], precisions, width, 
                        label='Precision', color='#3498db', alpha=0.8, edgecolor='black')
        bars5c = ax5.bar([i + width*0.5 for i in x], recalls, width, 
                        label='Recall', color='#e74c3c', alpha=0.8, edgecolor='black')
        bars5d = ax5.bar([i + width*1.5 for i in x], f1_scores, width, 
                        label='F1-Score', color='#f39c12', alpha=0.8, edgecolor='black')
        
        ax5.set_title('Comparativa Completa de Metricas', fontsize=14, fontweight='bold', pad=15)
        ax5.set_ylabel('Porcentaje (%)', fontsize=12)
        ax5.set_xticks(x)
        ax5.set_xticklabels(models, rotation=20, ha='right', fontsize=11)
        ax5.set_ylim(0, 105)
        ax5.legend(loc='lower right', fontsize=10, ncol=2)
        ax5.grid(axis='y', alpha=0.3, linestyle='--')

        # === GRÁFICA 6: Mejor Modelo ===
        best_model = max(results, key=lambda x: x['accuracy'])
        ax6 = fig.add_subplot(gs[2, 1])
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            best_model['accuracy'] * 100,
            best_model['precision'] * 100,
            best_model['recall'] * 100,
            best_model['f1_score'] * 100
        ]
        bars6 = ax6.bar(range(len(metrics)), values, 
                       color=['#27ae60', '#3498db', '#e74c3c', '#f39c12'], 
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        ax6.set_title(f'GANADOR: {best_model["model"]} (Nodo {best_model["rank"]})', 
                     fontsize=14, fontweight='bold', pad=15)
        ax6.set_ylabel('Porcentaje (%)', fontsize=12)
        ax6.set_xticks(range(len(metrics)))
        ax6.set_xticklabels(metrics, rotation=20, ha='right', fontsize=11)
        ax6.set_ylim(0, 105)
        ax6.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, val in enumerate(values):
            ax6.text(i, val + 1.5, f'{val:.2f}%', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        
        # Insertar en el frame
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# ==========================================
# LÓGICA DEL MAESTRO (RANK 0)
# ==========================================
class MasterGUI:
    def __init__(self, root, comm):
        self.root = root
        self.comm = comm
        self.root.title(f"Maestro MPI - Cluster de Control ({name})")
        self.root.geometry("1200x800") 
        self.root.configure(bg="#f0f0f0")
        self.results = None  # Almacenar resultados para ventana de métricas
        
        if size < 3:
            tk.Label(self.root, text=f"ERROR CRITICO: Se requieren 3 nodos.\nActual: {size}", 
                     fg="red", font=("Arial", 12, "bold")).pack(pady=20)
            return

        self.create_gui()

    def create_gui(self):
        # --- FRAME SUPERIOR ---
        top_frame = tk.Frame(self.root, bg="#2c3e50", bd=2, relief=tk.RAISED)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(top_frame, text="Sistema de ML Distribuido con MPI", 
                 font=("Arial", 18, "bold"), bg="#2c3e50", fg="white").pack(pady=8)
        
        tk.Label(top_frame, text="Arquitectura: Vectorizacion Centralizada + Compresion Optimizada", 
                 font=("Arial", 10), bg="#2c3e50", fg="#ecf0f1").pack(pady=2)

        # --- FRAME DE CONFIGURACIÓN ---
        config_frame = tk.LabelFrame(self.root, text="Configuracion de Hiperparametros", 
                                     font=("Arial", 12, "bold"), bg="#ecf0f1", bd=2, relief=tk.GROOVE)
        config_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=5, expand=False)

        notebook = ttk.Notebook(config_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- TAB 1: NAIVE BAYES ---
        nb_frame = tk.Frame(notebook, bg="white")
        notebook.add(nb_frame, text="Naive Bayes")
        
        tk.Label(nb_frame, text="Alpha (Suavizado):", bg="white", 
                font=("Arial", 10)).grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.nb_alpha = tk.Entry(nb_frame, width=15)
        self.nb_alpha.insert(0, "1.0")
        self.nb_alpha.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        tk.Label(nb_frame, text="Fit Prior:", bg="white", 
                font=("Arial", 10)).grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.nb_fit_prior = tk.BooleanVar(value=True)
        tk.Checkbutton(nb_frame, variable=self.nb_fit_prior, bg="white").grid(row=1, column=1, padx=10, pady=10, sticky="w")

        # --- TAB 2: RANDOM FOREST ---
        rf_frame = tk.Frame(notebook, bg="white")
        notebook.add(rf_frame, text="Random Forest")
        
        tk.Label(rf_frame, text="N de Arboles:", bg="white", 
                font=("Arial", 10)).grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.rf_n_estimators = tk.Entry(rf_frame, width=15)
        self.rf_n_estimators.insert(0, "100")
        self.rf_n_estimators.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        tk.Label(rf_frame, text="Profundidad Max:", bg="white", 
                font=("Arial", 10)).grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.rf_max_depth = tk.Entry(rf_frame, width=15)
        self.rf_max_depth.insert(0, "None")
        self.rf_max_depth.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        tk.Label(rf_frame, text="Min Samples Split:", bg="white", 
                font=("Arial", 10)).grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.rf_min_samples_split = tk.Entry(rf_frame, width=15)
        self.rf_min_samples_split.insert(0, "2")
        self.rf_min_samples_split.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        tk.Label(rf_frame, text="Criterio:", bg="white", 
                font=("Arial", 10)).grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.rf_criterion = ttk.Combobox(rf_frame, values=["gini", "entropy", "log_loss"], width=13)
        self.rf_criterion.set("gini")
        self.rf_criterion.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        # --- TAB 3: LOGISTIC REGRESSION ---
        lr_frame = tk.Frame(notebook, bg="white")
        notebook.add(lr_frame, text="Logistic Regression")
        
        tk.Label(lr_frame, text="Regularizacion (C):", bg="white", 
                font=("Arial", 10)).grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.lr_C = tk.Entry(lr_frame, width=15)
        self.lr_C.insert(0, "1.0")
        self.lr_C.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        tk.Label(lr_frame, text="Max Iteraciones:", bg="white", 
                font=("Arial", 10)).grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.lr_max_iter = tk.Entry(lr_frame, width=15)
        self.lr_max_iter.insert(0, "1000")
        self.lr_max_iter.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        tk.Label(lr_frame, text="Solver:", bg="white", 
                font=("Arial", 10)).grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.lr_solver = ttk.Combobox(lr_frame, values=["lbfgs", "liblinear", "saga", "sag"], width=13)
        self.lr_solver.set("lbfgs")
        self.lr_solver.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        tk.Label(lr_frame, text="Penalizacion:", bg="white", 
                font=("Arial", 10)).grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.lr_penalty = ttk.Combobox(lr_frame, values=["l2", "l1", "elasticnet", "none"], width=13)
        self.lr_penalty.set("l2")
        self.lr_penalty.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        # --- BOTONES DE CONTROL ---
        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(side=tk.TOP, pady=10)
        
        self.btn_start = tk.Button(btn_frame, text="DISTRIBUIR Y ENTRENAR", 
                                  command=self.start_thread, bg="#27ae60", fg="white", 
                                  font=("Arial", 12, "bold"), padx=20, pady=10, cursor="hand2")
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_metrics = tk.Button(btn_frame, text="VER METRICAS", 
                                     command=self.show_metrics, bg="#3498db", fg="white", 
                                     font=("Arial", 12, "bold"), padx=20, pady=10, 
                                     cursor="hand2", state=tk.DISABLED)
        self.btn_metrics.pack(side=tk.LEFT, padx=5)

        # --- FRAME DE LOGS ---
        log_frame = tk.LabelFrame(self.root, text="Registro de Eventos", 
                                 font=("Arial", 11, "bold"), bg="#ecf0f1", bd=2, relief=tk.GROOVE)
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_area = scrolledtext.ScrolledText(log_frame, height=15, bg="#2c3e50", 
                                                 fg="#ecf0f1", font=("Consolas", 9))
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- FRAME RESUMEN ---
        self.summary_frame = tk.LabelFrame(self.root, text="Resumen de Resultados", 
                                          font=("Arial", 11, "bold"), bg="white", bd=2, relief=tk.GROOVE)
        self.summary_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.summary_label = tk.Label(self.summary_frame, text="Esperando resultados...", 
                                     font=("Arial", 10), bg="white", fg="#7f8c8d", pady=15)
        self.summary_label.pack()

    def log(self, msg):
        timestamp = time.strftime('%H:%M:%S')
        self.log_area.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.log_area.see(tk.END)
        self.root.update_idletasks()

    def get_model_params(self):
        params = {
            'naive_bayes': {
                'alpha': float(self.nb_alpha.get()),
                'fit_prior': self.nb_fit_prior.get()
            },
            'random_forest': {
                'n_estimators': int(self.rf_n_estimators.get()),
                'max_depth': None if self.rf_max_depth.get() == "None" else int(self.rf_max_depth.get()),
                'min_samples_split': int(self.rf_min_samples_split.get()),
                'criterion': self.rf_criterion.get(),
                'n_jobs': 1
            },
            'logistic_regression': {
                'C': float(self.lr_C.get()),
                'max_iter': int(self.lr_max_iter.get()),
                'solver': self.lr_solver.get(),
                'penalty': self.lr_penalty.get()
            }
        }
        return params

    def start_thread(self):
        self.btn_start.config(state=tk.DISABLED, bg="#95a5a6")
        self.btn_metrics.config(state=tk.DISABLED)
        self.log_area.delete(1.0, tk.END)
        self.summary_label.config(text="Procesando...", fg="#7f8c8d")
        thread = threading.Thread(target=self.run_master_logic)
        thread.daemon = True
        thread.start()

    def show_metrics(self):
        if self.results:
            MetricsWindow(self.root, self.results)

    def update_summary(self, results):
        best = max(results, key=lambda x: x['accuracy'])
        summary_text = f"GANADOR: {best['model']} | Accuracy: {best['accuracy']:.2%} | " \
                      f"F1-Score: {best['f1_score']:.2%} | Tiempo: {best['time']:.3f}s | " \
                      f"Nodo: {best['node']} (Rank {best['rank']})"
        self.summary_label.config(text=summary_text, fg="#27ae60", font=("Arial", 11, "bold"))

    def run_master_logic(self):
        try:
            filepath = r"C:\MPIpractica\MPI-Distributed-ML-Classifier\sms+spam+collection\SMSSpamCollection"
            
            if not os.path.exists(filepath):
                self.log(f"ERROR: Dataset no encontrado")
                self.btn_start.config(state=tk.NORMAL, bg="#27ae60")
                return

            self.log("Cargando dataset...")
            df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
            X_train, X_test, y_train, y_test = train_test_split(
                df['message'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
            )
            self.log(f"Dataset: {len(df)} registros ({len(X_train)} train, {len(X_test)} test)")
            
            self.log("Vectorizando texto...")
            t_vec_start = time.time()
            vectorizer = CountVectorizer(max_features=3000, stop_words='english')
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            t_vec_end = time.time()
            self.log(f"Vectorizacion: {t_vec_end - t_vec_start:.3f}s ({X_train_vec.shape[1]} features)")
            
            model_params = self.get_model_params()
            self.log("Parametros configurados")
            
            data_packet = {
                'X_train_vec': X_train_vec,
                'X_test_vec': X_test_vec,
                'y_train': y_train.values,
                'y_test': y_test.values,
                'model_params': model_params
            }
            
            self.log("Comprimiendo datos...")
            t_compress_start = time.time()
            compressed_data = compress_data(data_packet)
            t_compress_end = time.time()
            
            original_size = len(pickle.dumps(data_packet, protocol=pickle.HIGHEST_PROTOCOL))
            compressed_size = len(compressed_data)
            ratio = (1 - compressed_size / original_size) * 100
            
            self.log(f"Compresion: {t_compress_end - t_compress_start:.3f}s "
                    f"({original_size/1024/1024:.2f} MB -> {compressed_size/1024/1024:.2f} MB, {ratio:.1f}%)")
            
            self.log(f"Distribuyendo a {size - 1} nodos...")
            t_bcast_start = time.time()
            self.comm.bcast(compressed_data, root=0)
            t_bcast_end = time.time()
            self.log(f"Datos distribuidos en {t_bcast_end - t_bcast_start:.4f}s")
            self.log("Esperando resultados...")

            results = []
            expected_slaves = min(size - 1, 3)
            
            for i in range(expected_slaves): 
                res = self.comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT)
                results.append(res)
                self.log(f"[{i+1}/{expected_slaves}] {res['model']}: Acc={res['accuracy']:.2%}, T={res['time']:.3f}s")

            if results:
                self.results = results
                self.log("\n" + "="*60)
                self.log("RESULTADOS FINALES")
                self.log("="*60)
                
                for res in sorted(results, key=lambda x: x['accuracy'], reverse=True):
                    self.log(f"{res['model']:20} | Acc: {res['accuracy']:.4f} | F1: {res['f1_score']:.4f}")
                
                best = max(results, key=lambda x: x['accuracy'])
                self.log(f"\nGANADOR: {best['model']} ({best['accuracy']:.2%})")
                self.log("="*60)
                
                self.root.after(0, lambda: self.update_summary(results))
                self.btn_metrics.config(state=tk.NORMAL)
            
        except Exception as e:
            self.log(f"ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.btn_start.config(state=tk.NORMAL, bg="#27ae60")

# ==========================================
# ESCLAVOS
# ==========================================
def run_slave_logic():
    try:
        compressed_data = comm.bcast(None, root=0)
        
        print(f"[Nodo {rank}] Descomprimiendo datos...")
        t_decompress_start = time.time()
        data = decompress_data(compressed_data)
        t_decompress_end = time.time()
        print(f"[Nodo {rank}] Descompresion: {t_decompress_end - t_decompress_start:.3f}s")
        
        X_train_vec = data['X_train_vec']
        X_test_vec = data['X_test_vec']
        y_train = data['y_train']
        y_test = data['y_test']
        model_params = data['model_params']

        if rank == 1:
            model_name = "Naive Bayes"
            params = model_params['naive_bayes']
            model = MultinomialNB(**params)
            
        elif rank == 2:
            model_name = "Random Forest"
            params = model_params['random_forest']
            model = RandomForestClassifier(**params)
            
        else:
            model_name = "Logistic Regression"
            params = model_params['logistic_regression']
            model = LogisticRegression(**params)

        print(f"[Nodo {rank}] ({name}): Entrenando {model_name}")

        start_t = time.time()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        end_t = time.time()
        
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='spam', average='binary')
        recall = recall_score(y_test, y_pred, pos_label='spam', average='binary')
        f1 = f1_score(y_test, y_pred, pos_label='spam', average='binary')
        total_time = end_t - start_t

        print(f"[Nodo {rank}] Completado en {total_time:.4f}s (Acc: {acc:.4f})")

        result = {
            'rank': rank,
            'node': name,
            'model': model_name,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'time': total_time
        }
        comm.send(result, dest=0, tag=TAG_RESULT)
        
    except Exception as e:
        print(f"[ERROR Nodo {rank}]: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    if rank == 0:
        root = tk.Tk()
        app = MasterGUI(root, comm)
        root.mainloop()
    else:
        run_slave_logic()