#Cinthya Sofía Cuéllar Hernández
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

# Busqueda
def busqueda_lineal(lista, x):
    for i, valor in enumerate(lista):
        if valor == x:
            return i
    return -1

def busqueda_binaria(lista, x):
    izquierda, derecha = 0, len(lista) - 1
    while izquierda <= derecha:
        medio = (izquierda + derecha) // 2
        if lista[medio] == x:
            return medio
        elif lista[medio] < x:
            izquierda = medio + 1
        else:
            derecha = medio - 1
    return -1

# lista ordenada
def generar_lista(tamaño):
    lista = np.random.randint(0, tamaño * 10, tamaño)
    return sorted(lista)

# Función de tiempo 
def medir_tiempo(funcion, lista, valor, repeticiones=5):
    total = 0
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        funcion(lista, valor)
        fin = time.perf_counter()
        total += (fin - inicio)
    return (total / repeticiones) * 1000 

#  GUI
class BusquedaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Comparación de Búsquedas")
        self.lista = []
        self.setup_gui()

    def setup_gui(self):
        #  tamaño#
        ttk.Label(self.root, text="Tamaño de lista:").grid(row=0, column=0, padx=5, pady=5)
        self.combo_tamaño = ttk.Combobox(self.root, values=[100, 1000, 10000, 100000])
        self.combo_tamaño.grid(row=0, column=1, padx=5, pady=5)

        #Generación de datos#
        self.btn_generar = ttk.Button(self.root, text="Generar datos", command=self.generar_datos)
        self.btn_generar.grid(row=0, column=2, padx=5, pady=5)

        # Entrada para el valor a buscar
        ttk.Label(self.root, text="Valor a buscar:").grid(row=1, column=0, padx=5, pady=5)
        self.entry_valor = ttk.Entry(self.root)
        self.entry_valor.grid(row=1, column=1, padx=5, pady=5)

        #búsqueda#
        ttk.Button(self.root, text="Búsqueda lineal", command=self.buscar_lineal).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(self.root, text="Búsqueda binaria", command=self.buscar_binaria).grid(row=2, column=1, padx=5, pady=5)

        #resultados#
        self.resultado = tk.StringVar()
        ttk.Label(self.root, textvariable=self.resultado).grid(row=3, column=0, columnspan=3, padx=5, pady=10)

        #gráfica#
        self.figura, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.figura, master=self.root)
        self.canvas.get_tk_widget().grid(row=4, column=0, columnspan=3, padx=5, pady=10)

    def generar_datos(self):
        try:
            tamaño = int(self.combo_tamaño.get())
            self.lista = generar_lista(tamaño)
            self.resultado.set(f"Lista generada con tamaño: {tamaño}")
        except:
            messagebox.showerror("Error", "Selecciona un tamaño válido")

    def buscar_lineal(self):
        self.buscar(busqueda_lineal, "Lineal")

    def buscar_binaria(self):
        self.buscar(busqueda_binaria, "Binaria")

    def buscar(self, algoritmo, nombre):
        if not self.lista:
            messagebox.showwarning("Advertencia", "Primero genera los datos")
            return
        try:
            valor = int(self.entry_valor.get())
        except:
            messagebox.showerror("Error", "Ingresa un número válido")
            return

        índice = algoritmo(self.lista, valor)
        tiempo = medir_tiempo(algoritmo, self.lista, valor)
        resultado = f"{nombre}: {'Encontrado en índice ' + str(índice) if índice != -1 else 'No encontrado'} | Tiempo: {tiempo:.3f} ms"
        self.resultado.set(resultado)
        self.actualizar_grafica()

    def actualizar_grafica(self):
        tamaños = [100, 1000, 10000, 100000]
        tiempos_lineal = [medir_tiempo(busqueda_lineal, generar_lista(t), -1) for t in tamaños]
        tiempos_binaria = [medir_tiempo(busqueda_binaria, generar_lista(t), -1) for t in tamaños]

        self.ax.clear()
        self.ax.plot(tamaños, tiempos_lineal, label="Lineal", marker='o')
        self.ax.plot(tamaños, tiempos_binaria, label="Binaria", marker='o')
        self.ax.set_title("Comparación de tiempos")
        self.ax.set_xlabel("Tamaño de lista")
        self.ax.set_ylabel("Tiempo (ms)")
        self.ax.legend()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = BusquedaGUI(root)
    root.mainloop()