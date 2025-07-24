import numpy as np

class SimulationState:
    def __init__(self, agent_ids):
        # Core per-agent storage
        self.num_agents = len(agent_ids)
        self.positions  = {i: None for i in agent_ids}  # p_i
        self.weights    = {i: 1.0  for i in agent_ids}  # w_i
        self.areas      = {i: None for i in agent_ids}  # |V_i|
        self.centroids  = {i: None for i in agent_ids}  # c_i
        self.bounding_polygon = None 
        
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
    

class UAV: 
    def __init__(self):
        self.weight = 23.0      # w_i
        self.Cd0 = 0.05            # Zero-lift drag coefficient
        self.rho = 1.225           # Air density
        self.S = 2.0                # Wing area
        self.e = 0.9                # Oswald efficiency factor
        self.AR = 17.0              # Aspect ratio
        self.m = 225.0                # Mass
        self.v_max = 22.0             # Maximum speed
        self.v_min = 12.0              # Minimum speed
        self.delta_v = 0.03
        self.A = 2 * self.m * self.m / (self.rho * np.pi * self.e * self.AR)
        self.B = 0.5 * self.rho * self.S * self.Cd0
