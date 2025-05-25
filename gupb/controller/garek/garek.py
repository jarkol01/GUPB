import random
from typing import Optional, Dict, List, Tuple

import networkx as nx
import numpy as np

from gupb import controller
from gupb.controller.garek.map import Map
from gupb.model import arenas, characters, coordinates, effects, tiles, weapons
import matplotlib.pyplot as plt

from gupb.model.characters import Facing
from gupb.model.coordinates import Coords
from gupb.model.tiles import TileDescription

POSSIBLE_ACTIONS = [
    characters.Action.TURN_LEFT,
    characters.Action.TURN_RIGHT,
    characters.Action.STEP_FORWARD,
    characters.Action.STEP_BACKWARD,
    characters.Action.STEP_LEFT,
    characters.Action.STEP_RIGHT,
    characters.Action.ATTACK,
]

FACING_STEP_MAP = {
    "opposite": characters.Action.STEP_BACKWARD,
    "turn_left": characters.Action.STEP_LEFT,
    "turn_right": characters.Action.STEP_RIGHT,
}

# Weapon preference (from best to worst)
WEAPON_PREFERENCE = {
    'amulet': 5,  # Gives better vision and attacks diagonally
    'axe': 4,  # Hits 3 tiles in front - good for endgame
    'sword': 3,
    'bow_loaded': 2,
    'bow_unloaded': 1,
    'knife': 0,  # Default weapon
}

ordinary_chaos_exploring_points = [
    Coords(5, 3),
    Coords(7, 10),
    Coords(2, 18),
    Coords(8, 18),
    Coords(16, 17),
    Coords(22, 9),
    Coords(13, 9),
]

MAP_EXPLORE_POINTS = {
    "ordinary_chaos": ordinary_chaos_exploring_points,
}


# noinspection PyUnusedLocal
# noinspection PyMethodMayBeStatic
class GarekController(controller.Controller):
    """
    G.A.R.E.K. - Game Agent Reinforcement Exploration Kernel
    """

    def __init__(self, first_name: str):
        self.first_name: str = first_name
        self.map: Optional[Map] = None  # initialized in reset
        self.menhir_position: Optional[Coords] = None
        self.mist_positions: List[Coords] = []
        self.last_health: int = 100
        self.current_health: int = 100
        self.current_weapon: str = 'knife'
        self.enemies_positions: Dict[str, Tuple[Coords, int]] = {}  # name -> (position, health)
        self.potions_positions: List[Coords] = []
        self.weapons_positions: Dict[Coords, str] = {}
        self.target_position: Optional[Coords] = None
        self.strategy: str = "explore"  # explore, find_weapon, find_potion, flee_mist, combat
        self.last_strategy_change_round: int = 0
        self.mist_detected: bool = False
        self.player_count: int = 0
        self.arena_name: str = ""
        self.round_counter: int = 0
        self.exploring_points: List[Coords] = []
        self.exploring_points_index: Optional[int] = None
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, GarekController):
            return self.first_name == other.first_name
        return False

    def __hash__(self) -> int:
        return hash(self.first_name)

    @property
    def name(self) -> str:
        return f'G.A.R.E.K. {self.first_name}'

    @property
    def preferred_tabard(self) -> characters.Tabard:
        return characters.Tabard.GAREK

    def praise(self, score: int) -> None:
        pass

    def reset(self, game_no: int, arena_description: arenas.ArenaDescription) -> None:
        self.map = Map(arena_description.name)
        self.arena_name = arena_description.name
        self.menhir_position = None
        self.mist_positions = []
        self.last_health = 100
        self.current_health = 100
        self.current_weapon = 'knife'
        self.enemies_positions = {}
        self.potions_positions = []
        self.weapons_positions = {}
        self.target_position = None
        self.strategy = "explore"
        self.mist_detected = False
        self.round_counter = 0
        self.exploring_points: List[Coords] = MAP_EXPLORE_POINTS[self.arena_name]
        self.exploring_points_index: Optional[int] = None
        self.original_graph = self.map.graph.copy()

    def decide(self, knowledge: characters.ChampionKnowledge) -> characters.Action:
        try:
            print(f"---------- Round {self.round_counter} ----------")
            self.round_counter += 1
            self.update_knowledge(knowledge)
            self.update_strategy(knowledge)
            print(f"Strategy: {self.strategy}")
            
            action = self.get_action_by_strategy(knowledge)
            print(f"Action: {action}")
        except Exception as e:
            print(f"Error: {e}")
            action = characters.Action.TURN_RIGHT
        return action

    def update_knowledge(self, knowledge: characters.ChampionKnowledge) -> None:
        """Update bot's knowledge about the game state."""
        self.player_count = knowledge.no_of_champions_alive
        self.current_health = self._get_health(knowledge)
        self.current_weapon = self._get_weapon_name(knowledge)
        
        # Update mist positions
        self.mist_positions = []
        for coord, tile in knowledge.visible_tiles.items():
            if hasattr(tile, 'effects') and tile.effects:
                for effect in tile.effects:
                    if effect.type == "mist":
                        self.mist_positions.append(coord)
                        self.mist_detected = True
        
        # Find menhir
        for coord, tile in knowledge.visible_tiles.items():
            if tile.type == "menhir":
                self.menhir_position = coord
        
        # Track enemies and update map to mark their positions as non-passable
        self.enemies_positions = {}
        for coord, tile in knowledge.visible_tiles.items():
            if tile.character and tile.character.controller_name != self.name:
                self.enemies_positions[tile.character.controller_name] = (coord, tile.character.health)
        
        # Update map by removing edges to enemy positions
        self._update_map_with_obstacles(self.enemies_positions.keys())
        
        # Track potions
        self.potions_positions = []
        for coord, tile in knowledge.visible_tiles.items():
            if tile.consumable:
                self.potions_positions.append(coord)
        
        # Track weapons
        for coord, tile in knowledge.visible_tiles.items():
            if tile.loot:
                self.weapons_positions[coord] = tile.loot.name

    def update_strategy(self, knowledge: characters.ChampionKnowledge) -> None:
        """Decide the best strategy based on current knowledge."""
        # if self.mist_detected and self.menhir_position:
        #     self.strategy = "flee_mist"
        #     self.target_position = self.menhir_position
        #     return
            
        # Low health: find a potion
        # if self.current_health <= 30 and self.potions_positions:
        #     self.strategy = "find_potion"
        #     self.target_position = self._get_nearest_position(knowledge.position, self.potions_positions)
        #     return
            
        # # Weapon upgrade: find a better weapon
        # current_weapon_value = WEAPON_PREFERENCE.get(self.current_weapon, 0)
        # best_weapon_position = None
        # best_weapon_value = current_weapon_value
        
        # for position, weapon_name in self.weapons_positions.items():
        #     weapon_value = WEAPON_PREFERENCE.get(weapon_name, 0)
        #     if weapon_value > best_weapon_value:
        #         best_weapon_value = weapon_value
        #         best_weapon_position = position
        
        # if best_weapon_position:
        #     self.strategy = "find_weapon"
        #     self.target_position = best_weapon_position
        #     return
            
        for enemy_name, (enemy_position, enemy_health) in self.enemies_positions.items():
            if self._should_attack_enemy(knowledge, enemy_position, enemy_health):
                self._change_strategy("combat")
                self.target_position = enemy_position
                return
            # elif self._should_flee(knowledge, enemy_position, enemy_health):
            #     self._change_strategy("flee")
        # Default: explore or move toward menhir for late game
        # if self.round_counter > 100 and self.menhir_position:
        #     self.strategy = "flee_mist"  # In late game, stay close to menhir
        #     self.target_position = self.menhir_position
        # else:
        #     self.strategy = "explore"
        self._change_strategy("explore")
    
    def _change_strategy(self, strategy: str) -> None:
        if strategy == self.strategy:
            self.last_strategy_change_round = self.round_counter
            return
        print(f"Actual strategy: {self.strategy}, new strategy: {strategy}")
        if strategy == "explore":
            if (self.round_counter - self.last_strategy_change_round) > 2:
                print("Changing strategy to explore")
                self.strategy = strategy
                self.last_strategy_change_round = self.round_counter
            return

        print(f"Changing strategy to {strategy}")
        self.strategy = strategy
        self.last_strategy_change_round = self.round_counter
            

    def get_action_by_strategy(self, knowledge: characters.ChampionKnowledge) -> characters.Action:
        """Get the best action based on current strategy."""
        if self.strategy == "explore":
            self._explore(knowledge)
        elif self.strategy == "combat" and (action := self._combat(knowledge)):
            return action
        # elif self.strategy == "flee":
        #     self._flee(knowledge)
        
        # For all strategies, pathfind to target
        path = self.map.get_path(knowledge.position, self.target_position)
        if len(path) > 1:
            next_position = path[1]
            prioritize_facing = self.strategy in ["explore", "combat"]
            
            return self._get_step_towards_target(next_position, knowledge, prioritize_facing)
                
        # If all else fails, turn and look for new options
        return characters.Action.TURN_RIGHT
    

    def _get_closest_exploration_point(self, position: Coords) -> int:
        min_distance = float('inf')
        min_index = 0

        for index, point in enumerate(self.exploring_points):
            distance = self.map.get_distance(position, point)
            if distance < min_distance:
                min_distance = distance
                min_index = index
        return min_index
        
    def _explore(self, knowledge: characters.ChampionKnowledge) -> None:
        if self.exploring_points_index is None:
            self.exploring_points_index = self._get_closest_exploration_point(knowledge.position)

        print(f"Exploring points: {self.exploring_points_index}")
        self.target_position = self.exploring_points[self.exploring_points_index % len(self.exploring_points)]

        if self.map.get_distance(knowledge.position, self.target_position) < 2:
            self.exploring_points_index += 1
        print(f"Exploring to {self.target_position}")

    def _combat(self, knowledge: characters.ChampionKnowledge) -> Optional[characters.Action]:
        facing = self._get_facing(knowledge)
        adjacent_positions = self._get_adjacent_positions(knowledge.position)
        distance_to_target = self.map.get_distance(knowledge.position, self.target_position)


        if self.current_weapon == "axe" and self.target_position in adjacent_positions:
            return characters.Action.ATTACK
        elif self.current_weapon == "sword" and distance_to_target <= 3 and self.target_position in self._get_facing_positions(knowledge.position, facing, 3):
            return characters.Action.ATTACK
        elif self.current_weapon == "bow" and distance_to_target <= 10 and self.target_position in self._get_facing_positions(knowledge.position, facing, 10):
            return characters.Action.ATTACK
        elif self.current_weapon == "amulet" and distance_to_target <= 3 and self.target_position in self._get_diagonal_positions(knowledge.position, 3):
            return characters.Action.ATTACK
        elif self.current_weapon == "knife" and self.target_position in self._get_facing_positions(knowledge.position, facing, 1):
            return characters.Action.ATTACK


    def _flee(self, knowledge: characters.ChampionKnowledge) -> Optional[characters.Action]:
        def get_valid_flee_target_away_from_enemies() -> Coords:
            # Get all possible nodes from the map
            possible_nodes = list(self.map.graph.nodes())
            random.shuffle(possible_nodes)  # Randomize order
            
            # For each enemy, filter out nodes that are too close or too far
            for enemy_position, enemy_health in self.enemies_positions.values():
                possible_nodes = [
                    node for node in possible_nodes 
                    if 3 <= self.map.get_distance(enemy_position, node) <= 7
                ]
                
            # Return first valid node, or current position if none found
            return possible_nodes[0] if possible_nodes else knowledge.position
        return
        self.target_position = get_valid_flee_target_away_from_enemies()


    def _get_step_towards_target(self, target_position, knowledge: characters.ChampionKnowledge, prioritize_facing: bool = False) -> characters.Action:
        """
        Get the action to step toward a target position.
        
        Args:
            target_position: The position to move towards
            knowledge: The champion's knowledge
            prioritize_facing: If True, prioritize facing the right direction over immediate movement
        """
        facing = self._get_facing(knowledge)
        
        for possible_facing in Facing:
            if knowledge.position + possible_facing.value == target_position:
                # If we're already facing that direction, step forward
                if facing == possible_facing:
                    return characters.Action.STEP_FORWARD
                
                # If we should prioritize facing the right direction
                if prioritize_facing:
                    # Turn to face the right direction first
                    if facing.turn_left() == possible_facing:
                        return characters.Action.TURN_LEFT
                    elif facing.turn_right() == possible_facing:
                        return characters.Action.TURN_RIGHT
                    elif facing.opposite() == possible_facing:
                        return characters.Action.TURN_RIGHT  # Turn right twice is faster than stepping backward
                
                # Otherwise use the most direct movement (sideways or backward if needed)
                for action in ["opposite", "turn_left", "turn_right"]:
                    if getattr(facing, action)() == possible_facing:
                        return FACING_STEP_MAP.get(action)
        
        # If we can't step directly (shouldn't happen with adjacent target), turn
        print(f"Turning right, because we can't step towards {target_position}")
        return characters.Action.TURN_RIGHT

    def _get_facing(self, knowledge: characters.ChampionKnowledge) -> Facing:
        """Get the current facing direction of the champion."""
        def is_our_champion(tile_description: TileDescription) -> bool:
            if not tile_description.character:
                return False
            return tile_description.character.controller_name == self.name

        tile_with_garek: TileDescription = list(filter(is_our_champion, knowledge.visible_tiles.values()))[0]
        return tile_with_garek.character.facing
    
    def _get_health(self, knowledge: characters.ChampionKnowledge) -> int:
        """Get the current health of the champion."""
        for tile in knowledge.visible_tiles.values():
            if tile.character and tile.character.controller_name == self.name:
                return tile.character.health
        return 0  # Shouldn't happen
    
    def _get_weapon_name(self, knowledge: characters.ChampionKnowledge) -> str:
        """Get the current weapon name of the champion."""
        for tile in knowledge.visible_tiles.values():
            if tile.character and tile.character.controller_name == self.name:
                return tile.character.weapon.name
        return "knife"  # Default
    
    def _get_nearest_position(self, from_pos: Coords, positions: List[Coords]) -> Optional[Coords]:
        """Get the nearest position from a list of positions."""
        if not positions:
            return None
        return min(positions, key=lambda pos: self._distance(from_pos, pos))
    
    def _distance(self, pos1: Coords, pos2: Coords) -> float:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)
    
    def _get_exploration_target(self, current_pos: Coords, center: Coords, radius: int) -> Coords:
        """Get a position to explore around a center point."""
        print(f"Exploring around {center} with radius {radius}")
        # Get a random angle
        angle = random.random() * 2 * 3.14159
        # Get a random distance (within radius)
        distance = random.randint(1, radius)
        # Calculate new position
        x = center.x + int(distance * np.cos(angle))
        y = center.y + int(distance * np.sin(angle))
        
        # Ensure the position is within the map
        try:
            # Try to find a valid position near the calculated one
            nearby_nodes = list(self.map.graph.nodes())
            valid_positions = [pos for pos in nearby_nodes 
                              if abs(pos.x - x) + abs(pos.y - y) < radius]
            if valid_positions:
                return min(valid_positions, 
                          key=lambda pos: abs(pos.x - x) + abs(pos.y - y))
        except Exception:
            pass
        
        # If all else fails, return the current position
        return current_pos
    
    def _get_adjacent_positions(self, position: Coords) -> List[Coords]:
        """Get all adjacent positions to the given position."""
        return [
            Coords(position.x + 1, position.y),
            Coords(position.x - 1, position.y),
            Coords(position.x, position.y + 1),
            Coords(position.x, position.y - 1)
        ]
    
    def _get_facing_positions(self, position: Coords, facing: Facing, distance: int) -> List[Coords]:
        positions = []
        for i in range(1, distance + 1):
            positions.append(position + facing.value * i)
        return positions
    
    def _get_diagonal_positions(self, position: Coords, distance: int = 3) -> List[Coords]:
        positions = []
        for i in range(1, distance + 1):
            positions.append(position + Facing.NORTH.value * i)
            positions.append(position + Facing.EAST.value * i)
            positions.append(position + Facing.SOUTH.value * i)
            positions.append(position + Facing.WEST.value * i)
        return positions
    def _should_attack_enemy(self, knowledge: characters.ChampionKnowledge, 
                           enemy_position: Coords, enemy_health: int) -> bool:
        if self.current_health < 3:
            print(f"Not attacking enemy, health: {self.current_health} < 3")
            return False
            
        health_advantage = self.current_health - enemy_health
        distance_to_enemy = self.map.get_distance(knowledge.position, enemy_position)

        score = (
            health_advantage * 15 -
            distance_to_enemy * 1
        )
        

        if self.player_count <= 1:
            attack_enemy = score > 0
            print(f"Attacking enemy: {attack_enemy} (health_advantage: {health_advantage}, distance_to_enemy: {distance_to_enemy}, score: {score})")
            return attack_enemy
        elif self.player_count <= 5:
            attack_enemy = score > 10
            print(f"Attacking enemy: {attack_enemy} (health_advantage: {health_advantage}, distance_to_enemy: {distance_to_enemy}, score: {score})")
            return attack_enemy
        else:
            attack_enemy = score > 20
            print(f"Attacking enemy: {attack_enemy} (health_advantage: {health_advantage}, distance_to_enemy: {distance_to_enemy}, score: {score})")
            return attack_enemy


    def _should_flee(self, knowledge: characters.ChampionKnowledge, enemy_position: Coords, enemy_health: int) -> bool:
        health_advantage = self.current_health - enemy_health
        distance_to_enemy = self.map.get_distance(knowledge.position, enemy_position)

        if distance_to_enemy < 3 and health_advantage < 0:
            return True
        return False


    def _update_map_with_obstacles(self, obstacles: List[Coords]) -> None:
        """Update the map by marking the given positions as non-passable."""
        # First, create a copy of the graph if we haven't done so already
        if not hasattr(self, 'original_graph'):
            self.original_graph = self.map.graph.copy()
        
        # Reset the graph to its original state
        self.map.graph = self.original_graph.copy()
        
        # Remove nodes for obstacles
        for obstacle in obstacles:
            if obstacle in self.map.graph:
                self.map.graph.remove_node(obstacle)
