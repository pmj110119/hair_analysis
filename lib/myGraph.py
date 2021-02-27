import numpy as np

class Edge:
    def __init__(self, source, destine, weight):
        self.weight = weight
        self.source = source
        self.destine = destine


class Graph:
    def __init__(self):
        self.vertices = set([])
        self.adjacents = {}
        self.inf = 99999.

    def add_startnode(self, vertice):
        self.vertices.add(vertice)
        self.adjacents[vertice] = {}

    def add_edge(self, edge):
        self.vertices.add(edge.source)
        self.vertices.add(edge.destine)
        if edge.source not in self.adjacents.keys():
            self.adjacents[edge.source] = {}
            self.adjacents[edge.source][edge.destine] = edge.weight
            if edge.destine not in self.adjacents.keys():
                self.adjacents[edge.destine] = {}
                self.adjacents[edge.destine][edge.source] = edge.weight
            else:
                self.adjacents[edge.destine][edge.source] = edge.weight
        else:
            self.adjacents[edge.source][edge.destine] = edge.weight
            if edge.destine not in self.adjacents.keys():
                self.adjacents[edge.destine] = {}
                self.adjacents[edge.destine][edge.source] = edge.weight
            else:
                self.adjacents[edge.destine][edge.source] = edge.weight
        # print("add edge from {} to {}, weight {}".format(edge.source, edge.destine, edge.weight))

    def delete_edge(self, source, destine):
        if source not in self.vertices or destine not in self.vertices:
            return False
        if destine in self.adjacents[source].keys():
            self.adjacents[source].pop(destine)

        if source in self.adjacents[destine].keys():
            self.adjacents[destine].pop(source)

    def get_adjacents(self, vertex):
        # print("get the adjacent vertices of vertex {}".format(vertex))
        if vertex not in self.adjacents.keys():
            return set([])
        return self.adjacents[vertex]

    def vertex_number(self):
        return len(self.vertices)

    def printgraph(self):
        for d in self.adjacents.keys():
            print("%d :" % d)
            for b in self.adjacents[d].keys():
                print(d, b, self.adjacents[d][b])

    def find_shortest_path(self, start, end):
        vertices = list(self.vertices)
        # 创建一个[节点号：序号]的字典
        vertices_dict = {}
        for vs,vnum in zip(vertices,range(len(vertices))):
            vertices_dict[vs]=vnum
        father = np.ones(len(vertices), dtype=int) * (-1)
        cost = np.ones(len(vertices)) * self.inf
        T = []
        T.append(end)

        cost[vertices_dict[end]] = 0.0
        now_node = end
        for i in range(len(vertices) - 1):
            for node in self.adjacents[now_node].keys():
                if node not in T:
                    if cost[vertices_dict[node]] > cost[vertices_dict[now_node]] + self.adjacents[now_node][node]:
                        cost[vertices_dict[node]] = cost[vertices_dict[now_node]] + self.adjacents[now_node][node]
                        father[vertices_dict[node]] = now_node
            min_cost = self.inf
            for j in vertices:
                if j not in T:
                    if min_cost > cost[vertices_dict[j]]:
                        min_cost = cost[vertices_dict[j]]
                        now_node = j
            T.append(now_node)
        shortest_path = []
        shortest_path.append(start)

        f = father[vertices_dict[start]]
        path_dist = self.adjacents[start][f]
        while f != -1:
            shortest_path.append(f)
            temp = father[vertices_dict[f]]
            if temp == -1:
                break
            path_dist += self.adjacents[f][temp]
            f = temp
        return shortest_path, path_dist

    def get_adjacents(self):
        return self.adjacents