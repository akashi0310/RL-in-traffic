"""
env/traffic_env.py
==================
SUMO Traffic Signal Control Environment (TraCI-based).

Network layout (intersection.net.xml):
  Incoming lanes : E0_0  (West→J8)  | -E0_0 (East→J8)
                   E1_0  (North→J8) | -E1_0 (South→J8)
  Traffic light  : J8  (4-phase static plan, overridden by RL agent)
  Phase 0 : N-S green  (state: rrrGGgrrrGGg)
  Phase 1 : N-S yellow (state: rrryyyrrryyy)
  Phase 2 : E-W green  (state: GGgrrrGGgrrr)
  Phase 3 : E-W yellow (state: yyyrrryyyrrr)

State (10-dim float32):
  [queue_E0, queue_-E0, queue_E1, queue_-E1,      ← halting vehicles / 25
   speed_E0, speed_-E0, speed_E1, speed_-E1,      ← mean speed / max speed
   current_green_phase,                            ← 0 or 1
   phase_time_norm]                                ← seconds / 60

Actions:
  0 → keep current green phase
  1 → switch to next green phase (yellow inserted automatically)

Reward: − Σ waiting_time across all incoming lanes (per sim-step sum)
"""

import os
import sys
import time
import subprocess
import numpy as np

# ── TraCI import ─────────────────────────────────────────────────────────────
if "SUMO_HOME" in os.environ:
    sys.path.insert(0, os.path.join(os.environ["SUMO_HOME"], "tools"))
try:
    import traci
except ImportError as e:
    raise ImportError(
        "TraCI not found. Install SUMO and set the SUMO_HOME environment variable.\n"
        "Download: https://sumo.dlr.de/docs/Downloads.php"
    ) from e


class TrafficEnv:
    # ── Constants ─────────────────────────────────────────────────────────────
    LANES = ["N2C_0", "N2C_1", "N2C_2", "E2C_0", "E2C_1", "E2C_2", "S2C_0", "S2C_1", "S2C_2", "W2C_0", "W2C_1", "W2C_2"]
    TL_ID = "center"
    GREEN_PHASES = [0, 4]          # indices of green phases in tlLogic
    YELLOW_DURATION = 3            # sim-seconds for yellow phase
    MIN_GREEN = 10                 # min green seconds before switch allowed
    STEP_SIZE = 5                  # sim-seconds per RL action step
    MAX_STEPS = 100                # 100 × 5 s = 500 s per episode

    def __init__(self, sumo_cfg: str, use_gui: bool = False, port: int = 8813):
        self.sumo_cfg = os.path.abspath(sumo_cfg)
        self.sumo_dir = os.path.dirname(self.sumo_cfg)
        self.use_gui = use_gui
        self.port = port
        self._connected = False
        self._step = 0
        self._phase_time = 0.0
        self._green_idx = 0        # 0 → GREEN_PHASES[0], 1 → GREEN_PHASES[1]

    # ── Size properties ───────────────────────────────────────────────────────
    @property
    def state_size(self) -> int:
        return len(self.LANES) * 2 + 2

    @property
    def action_size(self) -> int:
        return 2

    # ── Internal SUMO control ─────────────────────────────────────────────────
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

    # ── State & reward ────────────────────────────────────────────────────────
    def _get_state(self) -> np.ndarray:
        queues, speeds = [], []
        for lane in self.LANES:
            halt = traci.lane.getLastStepHaltingNumber(lane)
            spd = traci.lane.getLastStepMeanSpeed(lane)
            max_spd = traci.lane.getMaxSpeed(lane)
            queues.append(halt / 25.0)
            speeds.append(spd / max_spd if max_spd > 0 else 0.0)
        return np.array(
            queues + speeds + [self._green_idx, min(self._phase_time / 60.0, 1.0)],
            dtype=np.float32,
        )

    def _get_reward(self) -> float:
        # Using halting number (proxy for waiting time rate) is more stable than accumulated waiting time
        return -sum(traci.lane.getLastStepHaltingNumber(l) for l in self.LANES)

    # ── Gym-style API ─────────────────────────────────────────────────────────
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

        # 1. Optionally switch phases
        if action == 1 and self._phase_time >= self.MIN_GREEN:
            yellow = self.GREEN_PHASES[self._green_idx] + 1
            self._set_phase(yellow)
            for _ in range(self.YELLOW_DURATION):
                traci.simulationStep()
                reward += self._get_reward()  # Capture penalty during yellow

            self._green_idx = 1 - self._green_idx
            self._set_phase(self.GREEN_PHASES[self._green_idx])
            self._phase_time = 0.0

        # 2. Advance one RL timestep
        for _ in range(self.STEP_SIZE):
            traci.simulationStep()
            reward += self._get_reward()

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
        }

    def close(self):
        self._close()
