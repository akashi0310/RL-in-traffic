"""
env/traffic_env.py
==================
SUMO Traffic Signal Control Environment (TraCI-based).

4 lane groups (each group gets its own green phase):
  Phase 0 : N2C_0, N2C_1, S2C_0, S2C_1  (N-S straight + right)
  Phase 1 : E2C_0, E2C_1, W2C_0, W2C_1  (E-W straight + right)
  Phase 2 : N2C_2, S2C_2                 (N-S left turn)
  Phase 3 : E2C_2, W2C_2                 (E-W left turn)

State (10-dim float32):
  [queue_g0, queue_g1, queue_g2, queue_g3,         <- summed halting vehicles per group / 25
   speed_g0, speed_g1, speed_g2, speed_g3,         <- mean speed per group / max speed
   current_green_phase,                            <- 0..3
   phase_time_norm]                                <- seconds / 60

Actions (square layout: 0-1-2-3 as corners):
  0 -> stay at current green phase
  1 -> forward  (next phase, +1 mod 4)
  2 -> diagonal (opposite phase, +2 mod 4)
  3 -> backward (previous phase, +3 mod 4)

Reward: - sum halting vehicles across all lanes (per sim-step sum)

Scenarios (route files):
  normal_traffic      - Gaussian bell-curve, moderate volume
  crowded_all         - High constant volume from all directions
  crowded_single      - High volume N-S only (+ EW variant)
  crowded_fluctuate   - Alternating N-S / E-W heavy every 100s
"""

import os
import sys
import time
import random
import subprocess
import numpy as np

# -- TraCI import -------------------------------------------------------------
if "SUMO_HOME" in os.environ:
    sys.path.insert(0, os.path.join(os.environ["SUMO_HOME"], "tools"))
try:
    import traci
except ImportError as e:
    raise ImportError(
        "TraCI not found. Install SUMO and set the SUMO_HOME environment variable.\n"
        "Download: https://sumo.dlr.de/docs/Downloads.php"
    ) from e


# -- Available scenario route files -------------------------------------------
SCENARIOS = {
    "normal":     "normal_traffic.rou.xml",
    "crowded_all": "crowded_all.rou.xml",
    "crowded_ns":  "crowded_single.rou.xml",
    "crowded_ew":  "crowded_single_ew.rou.xml",
    "fluctuate":   "crowded_fluctuate.rou.xml",
    "bernoulli":   "bernoulli.rou.xml",
}


def compute_reward(lanes: list, left_lanes: list, mode: str = "wait") -> float:
    """
    Standardized reward function shared between Environment and Evaluation.
    
    Modes:
    - 'wait': Negative cumulative waiting time (seconds)
    - 'count': Negative number of halting vehicles
    """
    reward = 0.0
    for l in lanes:
        if mode == "wait":
            val = traci.lane.getWaitingTime(l)
        else:  # "count"
            val = traci.lane.getLastStepHaltingNumber(l)

        if l in left_lanes:
            reward -= val * 1.5
        else:
            reward -= val
    return reward


class TrafficEnv:
    # -- Constants -------------------------------------------------------------
    LANE_GROUPS = [["N2C_0", "N2C_1", "S2C_0", "S2C_1"],
                   ["E2C_0", "E2C_1", "W2C_0", "W2C_1"],
                   ["N2C_2", "S2C_2"],
                   ["E2C_2", "W2C_2"]]
    ALL_LANES = [l for group in LANE_GROUPS for l in group]
    TL_ID = "center"
    LEFT_LANES = ["N2C_2", "S2C_2", "E2C_2", "W2C_2"]
    NUM_PHASES = 4
    GREEN_PHASES = [0, 2, 4, 6]    # indices of green phases in tlLogic
    YELLOW_DURATION = 3            # sim-seconds for yellow phase
    MIN_GREEN = 10                 # min green seconds before switch allowed
    STEP_SIZE = 10                 # T=10 sim-seconds per RL action step
    MAX_STEPS = 50                 # 50 x 10 s = 500 s per episode
    GAMMA = 0.99                   # Discount factor for sim-steps

    def __init__(self, sumo_cfg: str, use_gui: bool = False, port: int = 8813,
                 scenarios: list = None, bernoulli_p: float = 0.05,
                 reward_mode: str = "wait"):
        """
        Parameters
        ----------
        sumo_cfg : str
            Path to the base .sumocfg file.
        use_gui : bool
            Launch sumo-gui instead of headless sumo.
        port : int
            TraCI port.
        scenarios : list of str or None
            Scenario names to sample from on each reset().
            Valid names: "normal", "crowded_all", "crowded_ns", "crowded_ew",
                         "fluctuate", "bernoulli", or "all" (uses every scenario).
            If None or empty, uses the route file from the .sumocfg as-is.
        bernoulli_p : float
            Spawning probability per lane per second for 'bernoulli' scenario.
        reward_mode : str
            'wait' (waiting time) or 'count' (halting vehicles).
        """
        self.sumo_cfg = os.path.abspath(sumo_cfg)
        self.sumo_dir = os.path.dirname(self.sumo_cfg)
        self.use_gui = use_gui
        self.port = port
        self.bernoulli_p = bernoulli_p
        self.reward_mode = reward_mode
        self._connected = False
        self._step = 0
        self._phase_time = 0.0
        self._green_idx = 0

        # Build list of route files to sample from
        self._route_files = []
        if scenarios:
            for s in scenarios:
                if s == "all":
                    self._route_files = list(SCENARIOS.values())
                    break
                if s not in SCENARIOS:
                    raise ValueError(
                        f"Unknown scenario '{s}'. "
                        f"Choose from: {list(SCENARIOS.keys())} or 'all'")
                self._route_files.append(SCENARIOS[s])

        self._current_scenario = None

    # -- Size properties -------------------------------------------------------
    @property
    def state_size(self) -> int:
        return self.NUM_PHASES * 2 + 2

    @property
    def action_size(self) -> int:
        return 4

    # -- Internal SUMO control -------------------------------------------------
    def _launch(self):
        binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [
            binary, "-c", self.sumo_cfg,
            "--remote-port", str(self.port),
            "--waiting-time-memory", "3600",
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--duration-log.disable", "true",
        ]
        # Override route file if using scenario mixing
        if self._route_files:
            route = random.choice(self._route_files)
            self._current_scenario = route
            
            # Dynamically generate Bernoulli routes if selected
            if route == "bernoulli.rou.xml":
                self._generate_bernoulli_routes()
                
            cmd += ["--route-files", route]

        if self.use_gui:
            cmd += ["--start", "--delay", "50"]
        self._proc = subprocess.Popen(
            cmd, cwd=self.sumo_dir,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)          # wait for SUMO to bind the port
        retries = 10
        for i in range(retries):
            try:
                traci.init(self.port)
                self._connected = True
                return
            except Exception as e:
                if i < retries - 1:
                    time.sleep(0.5)
                else:
                    raise e

    def _close(self):
        if self._connected:
            try:
                traci.close()
            except Exception:
                pass
            self._connected = False
        if hasattr(self, "_proc"):
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()

    def _set_phase(self, phase_idx: int):
        """Set a phase and freeze it (disable SUMO's automatic phase advance)."""
        traci.trafficlight.setPhase(self.TL_ID, phase_idx)
        traci.trafficlight.setPhaseDuration(self.TL_ID, 999_999)

    def _generate_bernoulli_routes(self):
        """Generate a Bernoulli distribution route file where each lane independently spawns vehicles."""
        filepath = os.path.join(self.sumo_dir, "bernoulli.rou.xml")
        end_time = self.MAX_STEPS * self.STEP_SIZE
        
        # Directions: from -> [straight, right, left]
        directions = {
            "W2C": ["C2E", "C2S", "C2N"],
            "E2C": ["C2W", "C2N", "C2S"],
            "N2C": ["C2S", "C2W", "C2E"],
            "S2C": ["C2N", "C2E", "C2W"]
        }
        
        with open(filepath, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<routes>\n')
            f.write('    <vType id="car" vClass="passenger" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="13.89" carFollowModel="Krauss"/>\n')
            f.write('    <vType id="truck" vClass="truck" accel="1.2" decel="2.5" sigma="0.5" length="10.0" maxSpeed="11.11" carFollowModel="Krauss"/>\n')
            f.write('    <vType id="bus" vClass="bus" accel="1.2" decel="2.5" sigma="0.5" length="12.0" maxSpeed="11.11" carFollowModel="Krauss"/>\n')
            f.write('    <vTypeDistribution id="mixed_traffic" vTypes="car truck bus" probabilities="0.6 0.2 0.2"/>\n\n')
            
            for src_edge, dests in directions.items():
                # Define routes
                f.write(f'    <route id="{src_edge}_s" edges="{src_edge} {dests[0]}"/>\n')
                f.write(f'    <route id="{src_edge}_r" edges="{src_edge} {dests[1]}"/>\n')
                f.write(f'    <route id="{src_edge}_l" edges="{src_edge} {dests[2]}"/>\n')
                
                # Distributions
                # 0 & 1: Straight and Right (approx 70/30 of the 85% total flow)
                f.write(f'    <routeDistribution id="{src_edge}_sr" routes="{src_edge}_s {src_edge}_r" probabilities="0.7 0.3"/>\n')
                
                # Flow for Lane 0 (Straight/Right)
                f.write(f'    <flow id="flow_{src_edge}_0" route="{src_edge}_sr" type="mixed_traffic" '
                        f'begin="0" end="{end_time}" probability="{self.bernoulli_p}" '
                        f'departLane="0" departSpeed="max"/>\n')
                # Flow for Lane 1 (Straight/Right)
                f.write(f'    <flow id="flow_{src_edge}_1" route="{src_edge}_sr" type="mixed_traffic" '
                        f'begin="0" end="{end_time}" probability="{self.bernoulli_p}" '
                        f'departLane="1" departSpeed="max"/>\n')
                # Flow for Lane 2 (Left Turn only)
                f.write(f'    <flow id="flow_{src_edge}_2" route="{src_edge}_l" type="mixed_traffic" '
                        f'begin="0" end="{end_time}" probability="{self.bernoulli_p}" '
                        f'departLane="2" departSpeed="max"/>\n')
            
            f.write('</routes>\n')

    # -- State & reward --------------------------------------------------------
    def _get_state(self) -> np.ndarray:
        queues, speeds = [], []
        for group in self.LANE_GROUPS:
            halt = sum(traci.lane.getLastStepHaltingNumber(l) for l in group)
            spd_sum, max_spd_sum = 0.0, 0.0
            for l in group:
                spd_sum += traci.lane.getLastStepMeanSpeed(l)
                max_spd_sum += traci.lane.getMaxSpeed(l)
            queues.append(halt / 25.0)
            speeds.append(spd_sum / max_spd_sum if max_spd_sum > 0 else 0.0)
        return np.array(
            queues + speeds + [self._green_idx, min(self._phase_time / 60.0, 1.0)],
            dtype=np.float32,
        )

    def _get_reward(self) -> float:
        return compute_reward(self.ALL_LANES, self.LEFT_LANES, mode=self.reward_mode)

    # -- Gym-style API ---------------------------------------------------------
    def reset(self) -> np.ndarray:
        if self._connected:
            self._close()
        self._step = 0
        self._phase_time = 0.0
        self._green_idx = 0
        self._launch()
        self._set_phase(self.GREEN_PHASES[self._green_idx])
        for _ in range(5):           # warm-up: let a few vehicles spawn
            traci.simulationStep()
        return self._get_state()

    def step(self, action: int):
        reward = 0.0

        if action != 0 and self._phase_time >= self.MIN_GREEN:
            yellow = self.GREEN_PHASES[self._green_idx] + 1
            self._set_phase(yellow)
            for t in range(self.YELLOW_DURATION):
                traci.simulationStep()
                # Apply step-level discounting: total_reward = sum(gamma^t * r_t)
                reward += (self.GAMMA ** t) * self._get_reward()

            self._green_idx = (self._green_idx + action) % self.NUM_PHASES
            self._set_phase(self.GREEN_PHASES[self._green_idx])
            self._phase_time = 0.0

        for t in range(self.STEP_SIZE):
            traci.simulationStep()
            # If a yellow light occurred, we continue the discount power index appropriately.
            # However, for simplicity and alignment with standard MDP discretization, 
            # we reset the discount index for the new main action duration or 
            # continue it if we consider the whole step() as one interval.
            # Using t as the index for the duration of this action phase:
            reward += (self.GAMMA ** t) * self._get_reward()

        self._phase_time += self.STEP_SIZE
        self._step += 1
        done = self._step >= self.MAX_STEPS

        if done:
            self._close()
            return np.zeros(self.state_size, dtype=np.float32), reward, True, {}

        return self._get_state(), reward, False, {
            "step": self._step,
            "phase": self._green_idx,
            "phase_time": self._phase_time,
            "scenario": self._current_scenario,
        }

    def close(self):
        self._close()
