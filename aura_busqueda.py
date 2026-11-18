import time
import heapq
from collections import deque
from aima.search import Problem


# Problema del agente inteligente AURA usando AIMA
class AURAProblem(Problem):

    def __init__(self):
        super().__init__("I", "G")

        self.graph = {
            "I": {"V": 1, "X": 2},
            "V": {"A": 2, "X": 3},
            "A": {"P": 1, "U": 3},
            "X": {"U": 1},
            "P": {"R": 1, "F": 3},
            "U": {"F": 2},
            "R": {"H": 2},
            "F": {"H": 2},
            "H": {"G": 1},
            "G": {}
        }

        # Heuristica admisible
        self.h = {
            "I": 8, "V": 7, "A": 5, "X": 6,
            "P": 4, "U": 5, "R": 3,
            "F": 3, "H": 1, "G": 0
        }

        # Heuristica no admisible
        self.h_no = self.h.copy()
        self.h_no["U"] = 7

    def actions(self, state):
        return list(self.graph[state].keys())

    def result(self, state, action):
        return action

    # Costo acumulado de cada movimiento
    def path_cost(self, c, state1, action, state2):
        return c + self.graph[state1][state2]


# Busqueda BFS
def bfs(p):
    cola = deque([(p.initial, [p.initial], 0)])
    visitados = set()
    expandidos = 0
    generados = 1

    while cola:
        estado, camino, costo = cola.popleft()

        if p.goal_test(estado):
            return camino, costo, expandidos, generados

        if estado not in visitados:
            visitados.add(estado)
            expandidos += 1

            for accion in p.actions(estado):
                nuevo = p.result(estado, accion)

                if nuevo not in visitados:
                    nuevo_costo = p.path_cost(
                        costo, estado, accion, nuevo
                    )
                    cola.append((
                        nuevo, camino + [nuevo], nuevo_costo
                    ))
                    generados += 1

    return None, None, expandidos, generados


# Busqueda DFS
def dfs(p):
    pila = [(p.initial, [p.initial], 0)]
    visitados = set()
    expandidos = 0
    generados = 1

    while pila:
        estado, camino, costo = pila.pop()

        if p.goal_test(estado):
            return camino, costo, expandidos, generados

        if estado not in visitados:
            visitados.add(estado)
            expandidos += 1

            for accion in reversed(p.actions(estado)):
                nuevo = p.result(estado, accion)

                if nuevo not in visitados:
                    nuevo_costo = p.path_cost(
                        costo, estado, accion, nuevo
                    )
                    pila.append((
                        nuevo, camino + [nuevo], nuevo_costo
                    ))
                    generados += 1

    return None, None, expandidos, generados


# Busqueda A*
def astar(p, heuristica):
    contador = 0
    cola = [(heuristica[p.initial], 0, contador,
             p.initial, [p.initial])]

    mejores = {}
    expandidos = 0
    generados = 1

    while cola:
        f, costo, _, estado, camino = heapq.heappop(cola)

        if estado in mejores and costo >= mejores[estado]:
            continue

        mejores[estado] = costo

        if p.goal_test(estado):
            return camino, costo, expandidos, generados

        expandidos += 1

        for accion in p.actions(estado):
            nuevo = p.result(estado, accion)
            nuevo_costo = p.path_cost(costo, estado, accion, nuevo)

            if nuevo not in mejores or nuevo_costo < mejores[nuevo]:
                contador += 1
                nuevo_f = nuevo_costo + heuristica[nuevo]

                heapq.heappush(
                    cola,
                    (nuevo_f, nuevo_costo, contador,
                     nuevo, camino + [nuevo])
                )
                generados += 1

    return None, None, expandidos, generados


# Factor de ramificacion efectivo
def factor_ramificacion(nodos, profundidad):
    if profundidad == 0:
        return 0

    bajo = 0
    alto = max(2, nodos)

    for _ in range(60):
        medio = (bajo + alto) / 2
        total = sum(medio ** i for i in range(profundidad + 1))

        if total < nodos:
            bajo = medio
        else:
            alto = medio

    return round((bajo + alto) / 2, 3)


# Ejecutar cada algoritmo 100 veces
def ejecutar(nombre, funcion):
    tiempos = []
    resultado = None

    for _ in range(100):
        inicio = time.perf_counter()
        resultado = funcion()
        tiempos.append(time.perf_counter() - inicio)

    camino, costo, expandidos, generados = resultado
    promedio = sum(tiempos) / len(tiempos)
    profundidad = len(camino) - 1 if camino else 0
    factor = factor_ramificacion(generados, profundidad)

    print("\n" + nombre)
    print("Camino:", camino)
    print("Costo:", costo)
    print("Nodos expandidos:", expandidos)
    print("Nodos generados:", generados)
    print("Tiempo promedio:", round(promedio, 8), "segundos")
    print("Factor de ramificacion:", factor)


# Programa principal
if __name__ == "__main__":
    problema = AURAProblem()

    print("================================")
    print(" AGENTE INTELIGENTE AURA")
    print("================================")
    print("Estado inicial:", problema.initial)
    print("Estado objetivo:", problema.goal)

    ejecutar("BFS - Busqueda en anchura",
             lambda: bfs(problema))

    ejecutar("DFS - Busqueda en profundidad",
             lambda: dfs(problema))

    ejecutar("A* - Heuristica admisible",
             lambda: astar(problema, problema.h))

    ejecutar("A* - Heuristica no admisible",
             lambda: astar(problema, problema.h_no))

    print("\nFin del programa.")
