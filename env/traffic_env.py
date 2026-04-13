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

Reward: - sum halting vehicles / waiting time across all lanes (per sim-step sum)

Traffic model:
  Bernoulli spawn per lane — each lane independently spawns a vehicle every
  simulation second with probability `bernoulli_p`. The route file is
  regenerated at every reset() based on the current probability.
"""

import os
import sys
import time
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


BERNOULLI_ROUTE_FILE = "bernoulli.rou.xml"
RED_TIME_HARD_LIMIT = 100
RED_TIME_SOFT_LIMIT = 80

def compute_reward(lanes: list, left_lanes: list, mode: str = "wait") -> float:
    """
    Standardized reward function shared between Environment and Evaluation.

    Parameters
    ----------
    lanes : list[str]
        All entry lanes to include in the reward.
    left_lanes : list[str]
        Subset of lanes that get the 1.5x left-turn starvation penalty.
    mode : str
        - 'wait'    : Negative waiting time of the lead vehicle (queue head)
                      on each lane.
        - 'count'   : Negative number of halting vehicles on the lane.
        - 'harmonic': Lead-vehicle waiting time + halting count combined.
    """
    reward = 0.0
    for l in lanes:
        if mode == "harmonic" or mode == "wait":
            veh_ids = traci.lane.getLastStepVehicleIDs(l)
            lead_wait = 0.0
            if veh_ids:
                lead = max(veh_ids, key=lambda v: traci.vehicle.getLanePosition(v))
                lead_wait = traci.vehicle.getWaitingTime(lead)
            if mode == "harmonic":
                val = lead_wait*np.exp((lead_wait - RED_TIME_HARD_LIMIT) / (RED_TIME_HARD_LIMIT - RED_TIME_SOFT_LIMIT)) + 2*traci.lane.getLastStepHaltingNumber(l)
            else:
                val = lead_wait
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

    SCENARIOS = ("uniform", "horizontal", "vertical", "alternate")

    def __init__(self, sumo_cfg: str, use_gui: bool = False, port: int = 8813,
                 bernoulli_p: float = 0.05, reward_mode: str = "wait",
                 eval_mode: bool = False, drain_step_cap: int = 200,
                 scenario: str = "uniform"):
        """
        Parameters
        ----------
        sumo_cfg : str
            Path to the base .sumocfg file.
        use_gui : bool
            Launch sumo-gui instead of headless sumo.
        port : int
            TraCI port.
        bernoulli_p : float
            Spawning probability per lane per second (Bernoulli trial).
        reward_mode : str
            'wait' (waiting time) or 'count' (halting vehicles).
        """
        self.sumo_cfg = os.path.abspath(sumo_cfg)
        self.sumo_dir = os.path.dirname(self.sumo_cfg)
        self.use_gui = use_gui
        self.port = port
        self.bernoulli_p = bernoulli_p
        self.reward_mode = reward_mode
        self.eval_mode = eval_mode
        self.drain_step_cap = drain_step_cap
        if scenario not in self.SCENARIOS:
            raise ValueError(f"scenario must be one of {self.SCENARIOS}, got {scenario!r}")
        self.scenario = scenario
        self._connected = False
        self._step = 0
        self._phase_time = 0.0
        self._green_idx = 0

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
        self._generate_bernoulli_routes()
        cmd = [
            binary, "-c", self.sumo_cfg,
            "--route-files", BERNOULLI_ROUTE_FILE,
            "--remote-port", str(self.port),
            "--waiting-time-memory", "3600",
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--duration-log.disable", "true",
        ]
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

    def _scenario_segments(self):
        """
        Return list of (begin, end, p_horiz, p_vert) tuples that define the
        per-direction Bernoulli probability over time for the current scenario.
        Horizontal = E2C/W2C; Vertical = N2C/S2C.
        """
        end_time = self.MAX_STEPS * self.STEP_SIZE
        hi = self.bernoulli_p
        lo = self.bernoulli_p / 2.0

        if self.scenario == "uniform":
            return [(0, end_time, hi, hi)]
        if self.scenario == "horizontal":
            return [(0, end_time, hi, lo)]
        if self.scenario == "vertical":
            return [(0, end_time, lo, hi)]
        if self.scenario == "alternate":
            n_seg = 4
            seg_len = end_time // n_seg
            segs = []
            for i in range(n_seg):
                begin = i * seg_len
                end = end_time if i == n_seg - 1 else (i + 1) * seg_len
                if i % 2 == 0:
                    segs.append((begin, end, hi, lo))
                else:
                    segs.append((begin, end, lo, hi))
            return segs
        raise ValueError(self.scenario)

    def _generate_bernoulli_routes(self):
        """Generate a Bernoulli distribution route file where each lane independently spawns vehicles."""
        filepath = os.path.join(self.sumo_dir, BERNOULLI_ROUTE_FILE)

        # Directions: from -> [straight, right, left]
        directions = {
            "W2C": ["C2E", "C2S", "C2N"],
            "E2C": ["C2W", "C2N", "C2S"],
            "N2C": ["C2S", "C2W", "C2E"],
            "S2C": ["C2N", "C2E", "C2W"]
        }
        horizontal = {"W2C", "E2C"}
        segments = self._scenario_segments()

        with open(filepath, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<routes>\n')
            f.write('    <vType id="car" vClass="passenger" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="13.89" carFollowModel="Krauss"/>\n')
            f.write('    <vType id="truck" vClass="truck" accel="1.2" decel="2.5" sigma="0.5" length="10.0" maxSpeed="11.11" carFollowModel="Krauss"/>\n')
            f.write('    <vType id="bus" vClass="bus" accel="1.2" decel="2.5" sigma="0.5" length="12.0" maxSpeed="11.11" carFollowModel="Krauss"/>\n')
            f.write('    <vTypeDistribution id="mixed_traffic" vTypes="car truck bus" probabilities="0.6 0.2 0.2"/>\n\n')

            for src_edge, dests in directions.items():
                f.write(f'    <route id="{src_edge}_s" edges="{src_edge} {dests[0]}"/>\n')
                f.write(f'    <route id="{src_edge}_r" edges="{src_edge} {dests[1]}"/>\n')
                f.write(f'    <route id="{src_edge}_l" edges="{src_edge} {dests[2]}"/>\n')
                f.write(f'    <routeDistribution id="{src_edge}_sr" routes="{src_edge}_s {src_edge}_r" probabilities="0.7 0.3"/>\n')

                for si, (begin, end, p_h, p_v) in enumerate(segments):
                    p = p_h if src_edge in horizontal else p_v
                    # Lane 0 (straight/right)
                    f.write(f'    <flow id="flow_{src_edge}_0_s{si}" route="{src_edge}_sr" type="mixed_traffic" '
                            f'begin="{begin}" end="{end}" probability="{p}" '
                            f'departLane="0" departSpeed="max"/>\n')
                    # Lane 1 (straight/right)
                    f.write(f'    <flow id="flow_{src_edge}_1_s{si}" route="{src_edge}_sr" type="mixed_traffic" '
                            f'begin="{begin}" end="{end}" probability="{p}" '
                            f'departLane="1" departSpeed="max"/>\n')
                    # Lane 2 (left)
                    f.write(f'    <flow id="flow_{src_edge}_2_s{si}" route="{src_edge}_l" type="mixed_traffic" '
                            f'begin="{begin}" end="{end}" probability="{p}" '
                            f'departLane="2" departSpeed="max"/>\n')

            f.write('</routes>\n')

    def set_scenario(self, scenario: str):
        if scenario not in self.SCENARIOS:
            raise ValueError(f"scenario must be one of {self.SCENARIOS}, got {scenario!r}")
        self.scenario = scenario

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
        self._is_yellow = False
        self._launch()
        self._set_phase(self.GREEN_PHASES[self._green_idx])
        for _ in range(5):           # warm-up: let a few vehicles spawn
            traci.simulationStep()
        return self._get_state()

    def step(self, action: int):
        reward = 0.0
        scoring = self._step < self.MAX_STEPS  # only accumulate reward up to MAX_STEPS

        if action != 0 and self._phase_time >= self.MIN_GREEN:
            yellow = self.GREEN_PHASES[self._green_idx] + 1
            self._set_phase(yellow)
            self._is_yellow = True
            for t in range(self.YELLOW_DURATION):
                traci.simulationStep()
                if scoring:
                    reward += (self.GAMMA ** t) * self._get_reward()

            self._green_idx = (self._green_idx + action) % self.NUM_PHASES
            self._set_phase(self.GREEN_PHASES[self._green_idx])
            self._is_yellow = False
            self._phase_time = 0.0

        for t in range(self.STEP_SIZE):
            traci.simulationStep()
            if scoring:
                reward += (self.GAMMA ** t) * self._get_reward()

        self._phase_time += self.STEP_SIZE
        self._step += 1

        if self.eval_mode:
            past_spawn = self._step >= self.MAX_STEPS
            cleared = past_spawn and traci.simulation.getMinExpectedNumber() == 0
            hit_cap = self._step >= self.MAX_STEPS + self.drain_step_cap
            done = cleared or hit_cap
        else:
            done = self._step >= self.MAX_STEPS

        if done:
            self._close()
            return np.zeros(self.state_size, dtype=np.float32), reward, True, {}

        return self._get_state(), reward, False, {
            "step": self._step,
            "phase": self._green_idx,
            "phase_time": self._phase_time,
        }

    def close(self):
        self._close()
