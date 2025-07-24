import time
import numpy as np

class SimulationState:
    def __init__(self, agent_ids):
        # Core per-agent storage
        self.num_agents = len(agent_ids)
        self.positions  = {i: None for i in agent_ids}  # p_i
        self.weights    = {i: 1.0  for i in agent_ids}  # w_i
        self.areas      = {i: None for i in agent_ids}  # |V_i|
        self.centroids  = {i: None for i in agent_ids}  # c_i
        self.bounding_polygon = None  # to be set externally
        
        # Communication graph (static neighbors)
        self.neighbours = {i: [] for i in agent_ids}     # neighbor_map
        
    # --- Methods to access/update these dictionaries follow ---
    def get_agent_state(self, agent_id):
        """Return (position, weight, area, centroid) for agent_id."""
        return (
            self.positions[agent_id],
            self.weights[agent_id],
            self.areas[agent_id],
            self.centroids[agent_id],
        )
        
    def get_agent_ids(self):
        """Return list of all agent IDs."""
        return list(self.positions.keys())
    
    def get_neighbours(self, agent_id):
        """Return list of neighbour IDs for agent_id."""
        return self.neighbours[agent_id]
    
    def get_agent_seed_and_weight(self, agent_id):
        """Return (position, weight) for agent_id."""
        return self.positions[agent_id], self.weights[agent_id]
    
    def get_agent_area(self, agent_id):
        """Return area for agent_id."""
        return self.areas[agent_id]

    def set_agent_position(self, agent_id, pos):
        self.positions[agent_id] = pos

    def set_agent_weight(self, agent_id, w):
        self.weights[agent_id] = w

    def set_agent_area_centroid(self, agent_id, area, centroid):
        self.areas[agent_id]     = area
        self.centroids[agent_id] = centroid
    
    def register_agent(self, aid, pos, weight=0.0):
        self.positions[aid]  = pos
        self.weights[aid]    = weight
        self.areas[aid]      = None
        self.centroids[aid]  = None
        self.neighbours[aid] = []


