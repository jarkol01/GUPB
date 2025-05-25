from typing import Tuple, List

import networkx as nx
from networkx import Graph

from gupb.model.arenas import Arena

from gupb.model.coordinates import Coords


class Map:
    def __init__(self, map_name: str) -> None:
        self.graph = self.load_graph(map_name)

    @staticmethod
    def get_neighbours_for_coords(coords: Coords, map_size: Tuple) -> List[Coords]:
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        neighbours = []
        for direction in directions:
            new_x = coords.x + direction[0]
            new_y = coords.y + direction[1]

            if 0 <= new_x < map_size[0] and 0 <= new_y < map_size[1]:
                neighbours.append(Coords(new_x, new_y))
        return neighbours

    def load_graph(self, map_name: str) -> Graph:
        area = Arena.load(map_name)
        walkable_terrain = dict(filter(lambda item: item[1].passable, area.terrain.items()))

        # TODO: use mist_radius in the future

        graph = nx.Graph()
        graph.add_nodes_from(walkable_terrain.keys())
        for coords in walkable_terrain.keys():
            graph.add_edges_from([(coords, neighbour) for neighbour in self.get_neighbours_for_coords(coords, area.size) if neighbour in walkable_terrain])

        return graph

    def get_path(self, source: Coords, destination: Coords) -> list[Coords]:
        return nx.shortest_path(self.graph, source, destination)

    def get_distance(self, source: Coords, destination: Coords) -> int:
        return nx.shortest_path_length(self.graph, source, destination)


