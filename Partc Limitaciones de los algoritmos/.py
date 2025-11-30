#Cuéllar Hernández Cinthya Sofía
import itertools
import math

# Clase usada para representar una ciudad con coordenadas
class Ciudad:
    def __init__(self, nombre, x, y):
        self.nombre = nombre
        self.x = x
        self.y = y
    
    def __str__(self):
        return self.nombre

# función para calcular la distancia entre dos ciudades   
# usé distancia euclidiana porque es más fácil de entender #La distancia euclidiana es la medida de la distancia en línea recta entre dos puntos en un espacio euclidiano, calculada como la raíz cuadrada de la suma de las diferencias al cuadrado de sus coordenadas
def calcular_distancia(ciudad1, ciudad2):
 
    dx = ciudad1.x - ciudad2.x
    dy = ciudad1.y - ciudad2.y
    return math.sqrt(dx*dx + dy*dy)

# función principal encargada de resolver el TSP (Travelling Salesman Problem)
def resolver_tsp(ciudades):
    print("=✩ PROBLEMA DEL VIAJERO ✩=")
    print(f"Ciudades a visitar: {[c.nombre for c in ciudades]}")
    print()
    
    # Generar todas las permutaciones posibles de rutas
    # Comienza desde la primera ciudad y regresa a ella
    num_ciudades = len(ciudades)
    indices = list(range(num_ciudades))
    
    # se elimina la primera ciudad de las permutaciones porque siempre comienza ahí
    permutaciones = list(itertools.permutations(indices[1:]))
    
    mejor_ruta = None
    mejor_distancia = float('inf')  # Inicia con un número muy grande
    
    print("Evaluando todas las rutas posibles...")
    print("-" * 50)
    
    contador = 0
    todas_las_rutas = []
    
    # Se prueban todas las permutaciones
    for perm in permutaciones:
        # Construir la ruta completa (empezar y terminar en ciudad 0)
        ruta = [0] + list(perm) + [0]
        contador += 1
        
        # Se calcula distancia total de esta ruta
        distancia_total = 0
        for i in range(len(ruta) - 1):
            ciudad_actual = ciudades[ruta[i]]
            ciudad_siguiente = ciudades[ruta[i + 1]]
            distancia_total += calcular_distancia(ciudad_actual, ciudad_siguiente)
        
        # SE guarda la  información de esta ruta
        info_ruta = {
            'numero': contador,
            'ruta': ruta.copy(),
            'distancia': distancia_total
        }
        todas_las_rutas.append(info_ruta)
        
        # Se verifica si es la mejor ruta hasta ahora
        if distancia_total < mejor_distancia:
            mejor_distancia = distancia_total
            mejor_ruta = ruta.copy()
            
            # Muestra que se encontró una mejor ruta
            nombres_ruta = [ciudades[i].nombre for i in ruta]
            print(f"Ruta más óptima encontrada: #{contador}: {nombres_ruta} - Distancia: {distancia_total:.2f}")
    
    print("-" * 50)
    print(f"Total de rutas evaluadas: {contador}")
    print()
    
    # Mostrar resultados finales
    print("=✩ RESULTADOS ✩=")
    print("LA MEJOR RUTA ENCONTRADA ES:")
    
    # Convertir índices a nombres de ciudades
    nombres_mejor_ruta = [ciudades[i].nombre for i in mejor_ruta]
    print(" -> ".join(nombres_mejor_ruta))
    
    print(f"DISTANCIA TOTAL: {mejor_distancia:.2f}")
    
    # Se muestran otras opciones evaluadas
    print("\n=== ALGUNAS OTRAS RUTAS EVALUADAS ===")
    # se ordenan las rutas para mostrar las mejores opciones
    todas_las_rutas.sort(key=lambda x: x['distancia'])
    
    for i in range(min(5, len(todas_las_rutas))):
        ruta_info = todas_las_rutas[i]
        nombres = [ciudades[idx].nombre for idx in ruta_info['ruta']]
        print(f"Ruta #{ruta_info['numero']}: {' -> '.join(nombres)} - Distancia: {ruta_info['distancia']:.2f}")

# Función para crear cuidades ejemplo
def crear_ejemplo_ciudades():
    # se crean 5 ciudades en posiciones aleatorias pero fijas para que siempre de el mismo resultado
    ciudades = [
        Ciudad("Casa", 0, 0),           # Punto de partida
        Ciudad("Trabajo", 3, 4),    # Coordenadas (3,4) - distancia 5 desde casa
        Ciudad("CUCEI", 6, 8),           # Coordenadas (6,8)
        Ciudad("Tienda de ropa", 1, 7),         # Coordenadas (1,7)  
        Ciudad("Centro de la cuidad", 8, 2)           # Coordenadas (8,2)
    ]
    return ciudades

# Programa principal
if __name__ == "__main__":
    print("PROBLEMA DEL VIAJERO - FUERZA BRUTA")
    print()
    
    # Crear las ciudades del problema
    mis_ciudades = crear_ejemplo_ciudades()
    
    # se muestra la información de las ciudades
    print("Información de las ciudades:")
    for i, ciudad in enumerate(mis_ciudades):
        print(f"  {i}. {ciudad.nombre} - Posición: ({ciudad.x}, {ciudad.y})")
    print()
    
    # Matriz de distancias
    print("Matriz de distancias entre ciudades:")
    print("     ", end="")
    for ciudad in mis_ciudades:
        print(f"{ciudad.nombre:12}", end="")
    print()
    
    for i, ciudad1 in enumerate(mis_ciudades):
        print(f"{ciudad1.nombre:5}", end="")
        for j, ciudad2 in enumerate(mis_ciudades):
            if i == j:
                print(f"{'0':12}", end="")
            else:
                distancia = calcular_distancia(ciudad1, ciudad2)
                print(f"{distancia:12.2f}", end="")
        print()
    print()
    
    # Resolver el problema
    resolver_tsp(mis_ciudades)
    
    print("\n--- Fin del programa ---")

Captura de pantalla 2025-11-2
