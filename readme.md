# Distributed NLP Training with MPI

This project implements a text classification system (Spam vs Ham) using distributed parallel computing with MPI.

## 📑 Table of Contents

- [🚀 Technologies](#-technologies)
- [📊 Dataset](#-dataset)
- [⚙️ Architecture](#️-architecture)
  - [Technical Features](#technical-features)
- [📁 Project Structure](#-project-structure)
- [🔧 Installation](#-installation)
- [📦 Execution](#-execution)
  - [Project Path](#project-path)
  - [Execution Commands](#execution-commands)
  - [mpiexec Command Format](#mpiexec-command-format)
  - [Important Notes](#important-notes)
- [🎯 Usage](#-usage)
- [📈 Results](#-results)
- [🔍 Additional Scripts](#-additional-scripts)
- [📝 System Requirements](#-system-requirements)
- [🐛 Troubleshooting](#-troubleshooting)

## 🚀 Technologies

- **Python 3.12**
- **MPI4Py** (Message Passing Interface) - v4.1.1
- **Scikit-Learn** (ML Models) - v1.8.0
- **Tkinter** (Results visualization on Master Node)
- **Matplotlib** - v3.10.8 (Metrics charts)
- **Pandas** - v2.3.3 (Data handling)
- **NumPy** - v2.4.0

## 📊 Dataset

The project uses the **SMS Spam Collection v.1**:
- **Total messages:** 5,574 SMS
- **Legitimate messages (ham):** 4,827 (86.6%)
- **Spam messages:** 747 (13.4%)
- **Format:** Text file with two columns (label, message)
- **Location:** `sms+spam+collection/SMSSpamCollection`

## ⚙️ Architecture

The system works with a **Master-Slave** scheme:

1. **Node 0 (Master):**
   - Loads and preprocesses the dataset
   - Vectorizes text using CountVectorizer (3000 features)
   - Compresses data for efficient transfer
   - Distributes data to worker nodes
   - Receives and visualizes results with graphical interface (Tkinter)
   - Displays detailed metrics with charts (Accuracy, F1-Score, Precision, Recall, Time)

2. **Node 1 (Worker):**
   - Trains a **Naive Bayes** model (MultinomialNB)
   - Configurable: alpha (smoothing), fit_prior

3. **Node 2 (Worker):**
   - Trains a **Random Forest** model
   - Configurable: n_estimators, max_depth, min_samples_split, criterion

4. **Node 3+ (Worker):**
   - Trains a **Logistic Regression** model
   - Configurable: C (regularization), max_iter, solver, penalty

### Technical Features

- **Data compression:** Pickle + Zlib for optimized transfer
- **Centralized vectorization:** Master vectorizes once and distributes
- **Graphical interface:** Hyperparameter configuration and results visualization
- **Complete metrics:** Accuracy, Precision, Recall, F1-Score, Training time

## 📁 Project Structure

```
MPI-Distributed-ML-Classifier/
├── practica5_mpi.py          # Main distributed system file
├── requirements.txt           # Project dependencies
├── readme.md                 # This file
├── sms+spam+collection/      # Dataset
│   ├── SMSSpamCollection     # Data file
│   └── readme                # Dataset documentation
└── other_scripts/            # MPI example and test scripts
    ├── hello.py              # Basic MPI example
    ├── test_mpi.py           # MPI test
    ├── detectar_nodos.py     # Node detection in cluster
    ├── broadcast.py          # Broadcast example
    ├── scatter.py            # Scatter example
    ├── gather.py             # Gather example
    ├── send.py               # Send/receive example
    ├── reduction.py          # Reduction operations example
    └── allToAll.py           # All-to-all communication example
```

## 🔧 Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure MPI:**
   - Make sure you have MPI installed on all nodes
   - On Windows, install Microsoft MPI (MS-MPI)
   - Run the MPI daemon on all nodes:
```bash
smpd -d
```

3. **Verify dataset path:**
   - The main file looks for the dataset at:
   ```
   C:\MPIpractica\MPI-Distributed-ML-Classifier\sms+spam+collection\SMSSpamCollection
   ```
   - If your path is different, modify line 419 in `practica5_mpi.py`

## 📦 Execution

### Project Path
```bash
cd C:\MPIpractica\MPI-Distributed-ML-Classifier
```

### Execution Commands

#### Local Execution (using hostfile)
```bash
mpiexec -n 3 -f hostfile python practica5_mpi.py
```

#### Distributed Execution
```bash
# Verify nodes
mpiexec -hosts 2 10.73.253.246 1 10.73.253.67 1 hostname

# Detect nodes
mpiexec -hosts 3 10.73.253.246 1 10.73.253.67 1 10.73.253.129 1 C:\Python312\python.exe C:\MPIpractica\MPI-Distributed-ML-Classifier\other_scripts\detectar_nodos.py

# Run main application
mpiexec -hosts 3 10.73.253.246 1 10.73.253.67 1 10.73.253.129 1 C:\Python312\python.exe C:\MPIpractica\MPI-Distributed-ML-Classifier\practica5_mpi.py
```

### mpiexec Command Format

```bash
mpiexec -hosts <num_hosts> <host1> <procs1> <host2> <procs2> ... <command>
```

Where:
- `num_hosts`: Number of different hosts
- `host`: IP or hostname
- `procs`: Number of processes on that host
- `command`: Python script to execute

### Important Notes

- **Listen on ALL machines:** Run `smpd -d` on each node before executing the application
- **Full path:** In distributed environments, use the full path to the Python executable and script
- **Minimum 3 nodes:** The system requires at least 3 processes (1 master + 2 workers) to function correctly

## 🎯 Usage

1. **Start the MPI daemon on all nodes:**
```bash
smpd -d
```

2. **Run the application:**
```bash
mpiexec -hosts 3 <host1> 1 <host2> 1 <host3> 1 python practica5_mpi.py
```

3. **On the Master Node (Rank 0):**
   - A graphical window will open with the control interface
   - Configure hyperparameters for each model
   - Click "DISTRIBUIR Y ENTRENAR" (DISTRIBUTE AND TRAIN)
   - View results and detailed metrics

4. **On Worker Nodes:**
   - Receive compressed data
   - Train their respective models
   - Send results to the master

## 📈 Results

The system provides:
- **Metrics per model:** Accuracy, Precision, Recall, F1-Score
- **Training time:** Performance comparison
- **Graphical visualization:** 6 comparative charts
- **Best model:** Automatic winner identification

## 🔍 Additional Scripts

In the `other_scripts/` folder you will find examples of MPI operations:
- `hello.py`: Basic greeting from each node
- `test_mpi.py`: Basic MPI communication test
- `detectar_nodos.py`: Detects and displays information from all nodes
- `broadcast.py`: Data broadcast example
- `scatter.py`: Data distribution example
- `gather.py`: Data collection example
- `send.py`: Point-to-point communication example
- `reduction.py`: Reduction operations example
- `allToAll.py`: All-to-all communication example

## 📝 System Requirements

- Python 3.12
- [Microsoft MPI for Windows](https://www.microsoft.com/en-us/download/details.aspx?id=100593)
- Network access between cluster nodes
- Permissions to execute remote processes

## 🐛 Troubleshooting

- **Connection error:** Verify that `smpd -d` is running on all nodes. It is also suggested to turn off the firewall and disable other network connections that are NOT WiFi (such as Virtualbox, WSL, VMWare, etc.)
- **Path not found:** Adjust the dataset path in `practica5_mpi.py` line 419
- **Import error:** Run `pip install -r requirements.txt`
- **Less than 3 nodes:** The system requires a minimum of 3 processes to function