def merge_sort(arr):
    if len(arr)>1:
        mid=len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        merge(arr, left_half, right_half)
    return arr

def merge(arr, left_half, right_half):
    i = j = k =0

    while i < len(left_half) and j < len(right_half):
        if left_half[i] < right_half[j]:
            arr[k]=left_half[i]
            i+=1
        else:
            arr[k] = right_half[j]
            j += 1
        k+=1

    while i < len(left_half):
        arr[k] = left_half[i]
        i += 1
        k += 1

    while j < len(right_half):
        arr[k] = right_half[j]
        j += 1
        k += 1

array = [7,38,27,43,3,9,82,10]
sorted_arr = merge_sort(array)
print("Arreglo ordenado:", sorted_arr)
mergeSort.py
Mostrando mergeSort.py.
[Practica ] Merge & QuickSort
JORGE ERNESTO LOPEZ ARCE DELGADO
•
1 oct
100 puntos
Fecha de entrega: 1 oct, 8:00
Implementar los códigos de la presentacion, subir a una carpeta (divideYvenceras) en su github.

Entregables:

Código(s).py 
captura del codigo funcional (que se vea código y funcionamiento)
link de Github de la carpeta (recuerden que la carpeta debe estar en un repositorio de la materia)
Tu trabajo
Entregado con retraso
Captura de pantalla 2025-10-01 183253.png
Imagen

Captura de pantalla 2025-10-01 183402.png
Imagen

mergeSort.py
Texto

quickSort.py
Texto

An-lisis-de-algoritmos/divideYvenceras at main · cinthyacuellar9179-gif/An-lisis-de-algoritmos · GitHub
https://github.com/cinthyacuellar9179-gif/An-lisis-de-algoritmos/blob/main/divideYvenceras

Comentarios privados

