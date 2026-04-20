"""
env/traffic_env.py
==================
SUMO Traffic Signal Control Environment (TraCI-based).

4 lane groups (each group gets its own green phase):
  Phase 0 : N2C_0, N2C_1, S2C_0, S2C_1  (N-S straight + right)
  Phase 1 : E2C_0, E2C_1, W2C_0, W2C_1  (E-W straight + right)
  Phase 2 : N2C_2, S2C_2                 (N-S left turn)
  Phase 3 : E2C_2, W2C_2                 (E-W left turn)

State (6-dim float32):
  [queue_g0, queue_g1, queue_g2, queue_g3,         <- summed halting vehicles per group / 25
   current_green_phase,                            <- 0..3
   phase_time_norm]                                <- seconds / 60

Actions:
  0 -> stay at current green phase
  1 -> next green phase (+1 mod 4)

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

import config

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
COUNT_REWARD_UPPER_BOUND = 10
WAIT_REWARD_UPPER_BOUND = 15
WAIT_TIME_LIMIT = 100

def compute_reward(lanes: list, left_lanes: list, return_components: bool = False) -> float | tuple[float, float, float]:
    """
    Standardized reward function based on the square of halting vehicles.
    """
    reward = 0.0
    count_part_total = 0.0

    for l in lanes:
        pending_count = len(traci.lane.getPendingVehicles(l))
        halting_count = traci.lane.getLastStepHaltingNumber(l) + pending_count
        count_val = (halting_count)** 2
        
        reward -= count_val
        count_part_total -= count_val

    reward = -np.sqrt(max(0, -reward))

    if return_components:
        return reward, 0.0, count_part_total
    return reward


class TrafficEnv:
    # -- Constants -------------------------------------------------------------
    LANE_GROUPS = [["N2C_0", "N2C_1", "S2C_0", "S2C_1"],
                   ["E2C_0", "E2C_1", "W2C_0", "W2C_1"],
                   ["N2C_2", "S2C_2"],
                   ["E2C_2", "W2C_2"]]
    ALL_LANES = [l for group in LANE_GROUPS for l in group]
    LEFT_LANES = ["N2C_2", "S2C_2", "E2C_2", "W2C_2"]
    TL_ID = "center"
    NUM_PHASES = 4
    GREEN_PHASES = [0, 2, 4, 6]    # indices of green phases in tlLogic

    SCENARIOS = ("uniform", "horizontal", "vertical", "alternate")

    def __init__(self, sumo_cfg: str | None = None, use_gui: bool = False, port: int = 8813,
                 bernoulli_p: float = 0.05,
                 eval_mode: bool = False, drain_step_cap: int = 200,
                 scenario: str = "uniform", max_steps: int = config.MAX_STEPS):
        """
        SUMO Traffic Signal Control Environment.
        """
        self.sumo_cfg = os.path.abspath(sumo_cfg or config.SUMO_CFG_PATH)
        self.sumo_dir = os.path.dirname(self.sumo_cfg)
        self.use_gui = use_gui
        self.port = port
        self.bernoulli_p = bernoulli_p
        self.eval_mode = eval_mode
        self.drain_step_cap = drain_step_cap
        self.max_steps = max_steps
        self.min_green = config.MIN_GREEN
        
        if scenario not in self.SCENARIOS:
            raise ValueError(f"scenario must be one of {self.SCENARIOS}, got {scenario!r}")
        self.scenario = scenario
        
        self._connected = False
        self._step = 0
        self._phase_time = 0.0
        self._green_idx = 0
        self._cum_departed = 0
        self._cum_arrived = 0

    @property
    def state_size(self) -> int:
        return self.NUM_PHASES + 2

    @property
    def action_size(self) -> int:
        return 2

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
        
        # Wait for SUMO to bind the port
        time.sleep(1.0)
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
        traci.trafficlight.setPhase(self.TL_ID, phase_idx)
        traci.trafficlight.setPhaseDuration(self.TL_ID, 999_999)

    def _scenario_segments(self):
        end_time = self.max_steps * config.STEP_SIZE
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
        filepath = os.path.join(self.sumo_dir, BERNOULLI_ROUTE_FILE)
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
                    for lane in range(3):
                        route = f"{src_edge}_sr" if lane < 2 else f"{src_edge}_l"
                        f.write(f'    <flow id="flow_{src_edge}_{lane}_s{si}" route="{route}" type="mixed_traffic" '
                                f'begin="{begin}" end="{end}" probability="{p}" '
                                f'departLane="{lane}" departSpeed="max"/>\n')
            f.write('</routes>\n')

    def set_scenario(self, scenario: str):
        if scenario not in self.SCENARIOS:
            raise ValueError(f"scenario must be one of {self.SCENARIOS}, got {scenario!r}")
        self.scenario = scenario

    def _get_state(self) -> np.ndarray:
        queues = []
        for group in self.LANE_GROUPS:
            # Include pending (backlogged) vehicles in the queue state
            halt = sum(traci.lane.getLastStepHaltingNumber(l) + len(traci.lane.getPendingVehicles(l)) for l in group)
            queues.append(halt / 25.0)
        return np.array(
            queues + [self._green_idx, min(self._phase_time / 60.0, 1.0)],
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        self._close()
        self._step = 0
        self._phase_time = 0.0
        self._green_idx = 0
        self._launch()
        self._set_phase(self.GREEN_PHASES[self._green_idx])
        # Short warm-up
        for _ in range(5):
            traci.simulationStep()
        return self._get_state()

    def step(self, action: int):
        # Action: 0=STAY, 1=NEXT
        switch_occurred = False
        if action == 1 and self._phase_time >= self.min_green:
            switch_occurred = True
            # Transition via yellow
            yellow = self.GREEN_PHASES[self._green_idx] + 1
            self._set_phase(yellow)
            for _ in range(config.YELLOW_DURATION):
                traci.simulationStep()
                self._cum_departed += traci.simulation.getDepartedNumber()
                self._cum_arrived += traci.simulation.getArrivedNumber()

            self._green_idx = (self._green_idx + 1) % self.NUM_PHASES
            self._set_phase(self.GREEN_PHASES[self._green_idx])
            self._phase_time = 0.0
        elif action == 1:
            # Revert to STAY if MIN_GREEN not met
            action = 0

        for _ in range(config.STEP_SIZE):
            traci.simulationStep()
            self._cum_departed += traci.simulation.getDepartedNumber()
            self._cum_arrived += traci.simulation.getArrivedNumber()

        # Calculate reward and its components
        reward, wait_sum, count_sum = compute_reward(
            self.ALL_LANES, self.LEFT_LANES, return_components=True
        )

        # Apply switching penalty to "let the agent know" that switching is costly
        if switch_occurred:
            reward -= config.SWITCH_PENALTY

        self._phase_time += config.STEP_SIZE
        self._step += 1

        curr_time = traci.simulation.getTime()
        if self.eval_mode:
            done = curr_time >= config.EVAL_DURATION
        else:
            done = self._step >= self.max_steps

        if done:
            info = {
                "cum_departed": self._cum_departed,
                "cum_arrived": self._cum_arrived
            }
            self._close()
            return np.zeros(self.state_size, dtype=np.float32), reward, True, info

        return self._get_state(), reward, False, {
            "step": self._step,
            "phase": self._green_idx,
            "phase_time": self._phase_time,
            "wait_part": wait_sum,
            "count_part": count_sum,
            "cum_departed": self._cum_departed,
            "cum_arrived": self._cum_arrived,
        }

    def close(self):
        self._close()
