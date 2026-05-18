"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

import math
from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeController(Controller):
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # For more info on the models, check out https://github.com/utiasDSL/drone-models
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]

        self.kp = np.array([0.4, 0.4, 1.25])    # 0.4, 0.4
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)
        self.g = 9.81

        # gates and obstacles tracking
        self._prev_gates_visited = obs["gates_visited"].copy()
        self._discovered_gates = {}
        self._discovered_obstacles = {}
        self._gate_adjusted_count = {}

        # Same waypoints as in the position controller. Determined by trial and error.
        waypoints = np.array(
            [
                [-1.5, 0.75, 0.05],
                [-1.0, 0.55, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.9, 0.7, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.1, 0.8],
                [-1.2, -0.1, 1.2],
                [-0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ]
        )

        self._nominal_waypoints = waypoints.copy()  # Store the nominal waypoints as fallback
        self._current_trajectory = waypoints.copy()  # Initialize with the default trajectory, can be updated based on gate discovery

        self._t_total = 13  # s
        t = np.linspace(0, self._t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(t, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()

        self._tick = 0
        self._finished = False

        # THROTTLING - damit wir nicht jeden Frame neu berechnen
        self._last_trajectory_update_tick = 0
        self._min_ticks_between_updates = 1  # ca. 0.2s bei 50Hz
        
        # OBSTACLE THROTTLING - separates throttling für Obstacles
        self._last_obstacle_update_tick = 0
        self._min_ticks_between_obstacle_updates = 20  # ca. 0.5s - länger halten!

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone."""
        
        #gemini:
        # Aktuelle Zeit berechnen
        t = self._tick / self._freq
    
        # Ende der Trajektorie prüfen
        if t >= self._t_total:
            t = self._t_total
            self._finished = True
        #t = min(self._tick / self._freq, self._t_total)
        #if t >= self._t_total:
        #    self._finished = True

        # ===== GATE DETECTION =====
        gates_visited = obs["gates_visited"]
        new_gates_detected = False
        for gate_idx in range(len(gates_visited)):
            if gates_visited[gate_idx] and gate_idx not in self._discovered_gates:
                self._discovered_gates[gate_idx] = obs["gates_pos"][gate_idx]
                new_gates_detected = True
                print(f"🟢 [TICK {self._tick}] Gate {gate_idx} ERKANNT at {obs['gates_pos'][gate_idx]}")

        # ===== OBSTACLE DETECTION =====
        obstacles_visited = obs["obstacles_visited"]
        for obstacle_idx in range(len(obstacles_visited)):
            if obstacles_visited[obstacle_idx] and obstacle_idx not in self._discovered_obstacles:
                self._discovered_obstacles[obstacle_idx] = obs["obstacles_pos"][obstacle_idx]
                print(f"🔴 [TICK {self._tick}] Obstacle {obstacle_idx} ERKANNT at {obs['obstacles_pos'][obstacle_idx]}")

        # ===== TRAJECTORY ADJUSTMENT =====
        if new_gates_detected:
            print(f"📍 Passe Trajektorie an (Throttle: {self._tick - self._last_trajectory_update_tick} ticks)")
            self._adjust_trajectory_near_gate(obs)

        # ===== OBSTACLE AVOIDANCE =====
        obstacle_avoided = self._check_obstacles_near_trajectory(obs)
        
        # ===== STANDARD PID =====
        des_pos = self._des_pos_spline(t)
        des_vel = self._des_vel_spline(t)
        des_yaw = 0.0

        pos_error = des_pos - obs["pos"]
        vel_error = des_vel - obs["vel"]

        self.i_error += pos_error * (1 / self._freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        target_thrust = np.zeros(3)
        target_thrust += self.kp * pos_error
        target_thrust += self.ki * self.i_error
        target_thrust += self.kd * vel_error
        target_thrust[2] += self.drone_mass * self.g

        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]
        thrust_desired = target_thrust.dot(z_axis)

        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        x_c_des = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)

        action = np.concatenate([euler_desired, [thrust_desired]], dtype=np.float32)

        self._prev_gates_visited = obs["gates_visited"].copy()

        return action
    
    def _adjust_trajectory_near_gate(self, obs: dict[str, NDArray[np.floating]]):
        """Ersetzt einen Wegpunkt durch drei Punkte (Entry, Center, Exit) am Gate."""
        if self._tick - self._last_trajectory_update_tick < self._min_ticks_between_updates:
            return
        
        self._last_trajectory_update_tick = self._tick
        
        # Finde das nächste Gate, das noch nicht "feinjustiert" wurde
        sorted_gates = sorted(self._discovered_gates.keys())
        next_gate_idx = None
        for g_idx in sorted_gates:
            if self._gate_adjusted_count.get(g_idx, 0) < 1:
                next_gate_idx = g_idx
                break
                
        if next_gate_idx is None:
            return

        # Gate-Daten extrahieren
        gate_pos = self._discovered_gates[next_gate_idx]
        gate_quat = obs["gates_quat"][next_gate_idx]
        gate_rot = R.from_quat(gate_quat)
        # Die X-Achse (Spalte 0) zeigt meistens durch das Gate
        gate_forward = gate_rot.as_matrix()[:, 0] 

        # --- 3-PUNKT PROFIL BERECHNEN ---
        offset = 0.15  # 20cm vor und nach dem Gate
        p_entry = gate_pos - gate_forward * offset
        p_center = gate_pos
        p_exit = gate_pos + gate_forward * offset

        # Finde den nächstgelegenen existierenden Wegpunkt in der aktuellen Trajektorie
        distances = np.linalg.norm(self._current_trajectory - gate_pos, axis=1)
        nearest_wp_idx = np.argmin(distances)

        # Logik: Wir entfernen den alten, ungenauen Wegpunkt und fügen unsere 3 präzisen Punkte ein
        traj_list = list(self._current_trajectory)
        
        # Ersetze den Punkt am Index nearest_wp_idx durch die 3 neuen Punkte
        traj_list[nearest_wp_idx : nearest_wp_idx + 1] = [p_entry, p_center, p_exit]
        
        self._current_trajectory = np.array(traj_list)
        self._gate_adjusted_count[next_gate_idx] = 1
        
        print(f"✅ Gate {next_gate_idx} optimiert: 3 Wegpunkte eingefügt.")
        self._update_spline()

    def _adjust_trajectory_near_gate_letzter_Stand(self, obs: dict[str, NDArray[np.floating]]):
        """Adjust trajectory by steering through gate openings, not through centers."""
        
        # Throttling check
        if self._tick - self._last_trajectory_update_tick < self._min_ticks_between_updates:
            return
        
        self._last_trajectory_update_tick = self._tick
        
        # Behandle die nächste UNDISKUTIERTE Gate
        sorted_gates = sorted(self._discovered_gates.keys())
        next_gate_idx = None
        for gate_idx in sorted_gates:
            if not hasattr(self, '_gate_adjusted_count'):
                self._gate_adjusted_count = {}
            if self._gate_adjusted_count.get(gate_idx, 0) < 2:
                next_gate_idx = gate_idx
                break
        
        if next_gate_idx is None:
            return
        
        gate_pos = self._discovered_gates[next_gate_idx]
        gate_quat = obs["gates_quat"][next_gate_idx]  # ← Gate-Orientierung
        
        # ===== WICHTIG: Berechne die Richtung durch die Gate-Öffnung =====
        # Die Gate-Öffnung zeigt in die X-Richtung des Gate-Frames
        # Wandle Quaternion zu Rotationsmatrix um
        from scipy.spatial.transform import Rotation as R
        gate_rot = R.from_quat(gate_quat)
        gate_forward = gate_rot.as_matrix()[:, 0]  # X-Achse des Gate-Frames (Öffnungsrichtung)
        
        print(f"  📌 Gate {next_gate_idx}: pos={gate_pos}, quat={gate_quat}")
        print(f"     Gate-Öffnungsrichtung: {gate_forward}")
        
        # Finde nächsten Waypoint
        distances = np.linalg.norm(self._current_trajectory - gate_pos, axis=1)
        nearest_wp_idx = np.argmin(distances)
        nearest_distance = distances[nearest_wp_idx]
        
        print(f"     nearest_wp={nearest_wp_idx}, dist={nearest_distance:.3f}m")
        
        # ===== WICHTIG: Verschiebe Waypoint zu Gate-Position + OFFSET in Öffnungsrichtung =====
        # Offset = 0.3m in die Richtung der Öffnung (damit Drohne DURCH die Gate fliegt)
        gate_waypoint = gate_pos + gate_forward * 0.25  # 0.25m in die Öffnungsrichtung
        
        old_wp = self._current_trajectory[nearest_wp_idx].copy()
        self._current_trajectory[nearest_wp_idx] = gate_waypoint.copy()
        
        print(f"  🔧 Waypoint {nearest_wp_idx} ERSETZT:")
        print(f"     Alt:          {old_wp}")
        print(f"     Gate-Center:  {gate_pos}")
        print(f"     Neu (offset): {self._current_trajectory[nearest_wp_idx]}")
        
        # Zähle Anpassungen
        if not hasattr(self, '_gate_adjusted_count'):
            self._gate_adjusted_count = {}
        self._gate_adjusted_count[next_gate_idx] = self._gate_adjusted_count.get(next_gate_idx, 0) + 1
        
        # Neuberechne Spline
        self._update_spline()
    
    def _adjust_trajectory_near_gate_old2(self, obs: dict[str, NDArray[np.floating]]):
        """Adjust trajectory AGGRESSIVELY when a new gate is discovered."""
        
        # Throttling check
        if self._tick - self._last_trajectory_update_tick < self._min_ticks_between_updates:
            return
        
        self._last_trajectory_update_tick = self._tick
        
        # Behandle die nächste UNDISKUTIERTE Gate (die mit dem kleinsten Index)
        sorted_gates = sorted(self._discovered_gates.keys())
        next_gate_idx = None
        for gate_idx in sorted_gates:
            # Finde die erste Gate, deren Waypoint nicht schon zu oft angepasst wurde
            if not hasattr(self, '_gate_adjusted_count'):
                self._gate_adjusted_count = {}
            if self._gate_adjusted_count.get(gate_idx, 0) < 2:  # Max 2x anpassen pro Gate
                next_gate_idx = gate_idx
                break
        
        if next_gate_idx is None:
            return
        
        gate_pos = self._discovered_gates[next_gate_idx]
        
        # Finde nächsten Waypoint
        distances = np.linalg.norm(self._current_trajectory - gate_pos, axis=1)
        nearest_wp_idx = np.argmin(distances)
        nearest_distance = distances[nearest_wp_idx]
        
        print(f"  📌 Gate {next_gate_idx}: pos={gate_pos}")
        print(f"     nearest_wp={nearest_wp_idx}, dist={nearest_distance:.3f}m")
        
        # AGGRESSIV: Verschiebe den Waypoint IMMER wenn Gate erkannt wird
        # Nicht abbrechen wenn "nah genug"!
        old_wp = self._current_trajectory[nearest_wp_idx].copy()
        self._current_trajectory[nearest_wp_idx] = gate_pos.copy()
        
        print(f"  🔧 Waypoint {nearest_wp_idx} ERSETZT mit Gate-Position:")
        print(f"     Alt:  {old_wp}")
        print(f"     Neu:  {self._current_trajectory[nearest_wp_idx]}")
        
        # Zähle wie oft diese Gate angepasst wurde
        if not hasattr(self, '_gate_adjusted_count'):
            self._gate_adjusted_count = {}
        self._gate_adjusted_count[next_gate_idx] = self._gate_adjusted_count.get(next_gate_idx, 0) + 1
        
        # Neuberechne Spline sofort
        self._update_spline()

    def _adjust_trajectory_near_gate_old(self, obs: dict[str, NDArray[np.floating]]):
        """Adjust trajectory when a new gate is discovered."""
        
        # Throttling check - WICHTIG: verhindert zu häufige Anpassungen
        if self._tick - self._last_trajectory_update_tick < self._min_ticks_between_updates:
            return
        
        self._last_trajectory_update_tick = self._tick
        
        # Nur die ERSTE (nächste) Gate anpassen, nicht alle!
        if len(self._discovered_gates) > 0:
            # Finde die Gate mit dem kleinsten Index (nächste Gate)
            gate_idx = min(self._discovered_gates.keys())
            gate_pos = self._discovered_gates[gate_idx]
            
            print(f"[DEBUG] Adjusting trajectory for gate {gate_idx} at pos {gate_pos}")
            
            # Finde nächsten Waypoint zur Gate-Position
            distances = np.linalg.norm(self._current_trajectory - gate_pos, axis=1)
            nearest_wp_idx = np.argmin(distances)
            nearest_distance = distances[nearest_wp_idx]
            
            print(f"[DEBUG] Nearest waypoint: idx={nearest_wp_idx}, dist={nearest_distance:.3f}m")
            
            # Wenn die Gate schon nah genug am Waypoint ist, nicht verschieben!
            if nearest_distance < 0.1:
                print(f"[DEBUG] Gate ist nah genug, keine Anpassung nötig")
                return
            
            # Verschiebe nur den EINEN nächsten Waypoint (nicht ±1 Nachbarn!)
            adjustment = gate_pos - self._current_trajectory[nearest_wp_idx]
            max_shift = 0.2  # Maximum 0.2m Verschiebung pro Anpassung
            adjustment_limited = adjustment * min(1.0, max_shift / np.linalg.norm(adjustment))
            
            print(f"[DEBUG] Adjusting waypoint {nearest_wp_idx} by {np.linalg.norm(adjustment_limited):.3f}m")
            self._current_trajectory[nearest_wp_idx] += adjustment_limited
            
            # Neuberechne Spline
            self._update_spline()

    def _update_spline(self): #gemini
        """Berechnet den Spline basierend auf der Distanz zwischen den Wegpunkten."""
        if len(self._current_trajectory) < 2:
            return

        # 1. Berechne die Abstände zwischen aufeinanderfolgenden Wegpunkten
        diffs = np.diff(self._current_trajectory, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        
        # 2. Erstelle einen kumulativen Distanz-Vektor (Bogenlänge)
        # Beispiel: [0, 0.5m, 1.2m, 2.0m, ...]
        s = np.zeros(len(self._current_trajectory))
        s[1:] = np.cumsum(dists)
        
        # 3. Skaliere die Distanz auf die Gesamtzeit self._t_total
        # So bleibt die Durchschnittsgeschwindigkeit konstant, egal wie viele Punkte wir einfügen
        t_scaled = (s / s[-1]) * self._t_total if s[-1] > 0 else np.linspace(0, self._t_total, len(s))

        try:
            self._des_pos_spline = CubicSpline(t_scaled, self._current_trajectory)
            self._des_vel_spline = self._des_pos_spline.derivative()
            # Wir speichern die max Distanz für das Clipping in compute_control
            self._max_s = s[-1] 
        except Exception as e:
            print(f"❌ Spline-Fehler: {e}")
            self._reset_to_nominal_trajectory()

    def _update_spline_old(self):
        """Update the cubic spline based on the current trajectory waypoints."""
        try:
            t = np.linspace(0, self._t_total, len(self._current_trajectory))
            self._des_pos_spline = CubicSpline(t, self._current_trajectory)
            self._des_vel_spline = self._des_pos_spline.derivative()
        except Exception as e:
            print(f"error with updating spline: {e}")
            self._reset_to_nominal_trajectory()
    
    def _check_obstacles_near_trajectory(self, obs: dict[str, NDArray[np.floating]]):
        """Check for obstacles near the current trajectory and adjust if necessary."""
        
        # ===== THROTTLING: Nicht jeden Frame! =====
        if self._tick - self._last_obstacle_update_tick < self._min_ticks_between_obstacle_updates:
            return False  # Skip diese Prüfung
        
        self._last_obstacle_update_tick = self._tick
        
        trajectory_adjusted = False
        
        # Iterate over all obstacles
        for obs_idx in range(len(obs["obstacles_pos"])):
            # Skip if not visited
            if not obs["obstacles_visited"][obs_idx]:
                continue
            
            obstacle_pos = obs["obstacles_pos"][obs_idx]
            
            # Compute distance from obstacle to trajectory (XY only)
            wp_xy = self._current_trajectory[:, :2]
            obs_xy = obstacle_pos[:2]
            
            distances = np.linalg.norm(wp_xy - obs_xy, axis=1)
            min_distance = np.min(distances)
            
            COLLISION_THRESHOLD = 0.2
            if min_distance < COLLISION_THRESHOLD:
                print(f"⚠️ [TICK {self._tick}] Obstacle {obs_idx} zu nah: {min_distance:.3f}m")
                print(f"   Pos: {obstacle_pos}")
                
                AFFECTED_RADIUS = 0.6
                affected_indices = np.where(distances < AFFECTED_RADIUS)[0]
                
                if len(affected_indices) > 0:
                    print(f"   → Verschiebe {len(affected_indices)} Waypoints: {list(affected_indices)}")
                    
                    for wp_idx in affected_indices:
                        direction = self._current_trajectory[wp_idx, :2] - obs_xy
                        direction_norm = np.linalg.norm(direction)
                        
                        if direction_norm > 0.01:
                            direction_normalized = direction / direction_norm
                            SHIFT_DISTANCE = 0.25  # Weniger aggressiv (0.3 → 0.25)
                            shift = direction_normalized * SHIFT_DISTANCE
                            
                            old_pos = self._current_trajectory[wp_idx, :2].copy()
                            self._current_trajectory[wp_idx, 0] += shift[0]
                            self._current_trajectory[wp_idx, 1] += shift[1]
                            
                            print(f"     Wp {wp_idx}: {old_pos} → {self._current_trajectory[wp_idx, :2]}")
                    
                    trajectory_adjusted = True
        
        if trajectory_adjusted:
            # Sicherheitscheck: Stelle sicher dass Gates noch erreichbar sind
            for gate_idx, gate_pos in self._discovered_gates.items():
                gate_quat = obs["gates_quat"][gate_idx]
                gate_rot = R.from_quat(gate_quat)
                gate_forward = gate_rot.as_matrix()[:, 0]
                gate_waypoint = gate_pos + gate_forward * 0.25
                
                distances_to_gate = np.linalg.norm(self._current_trajectory - gate_waypoint, axis=1)
                nearest_dist = np.min(distances_to_gate)
                
                if nearest_dist > 1.0:
                    print(f"⚠️ WARNUNG: Gate {gate_idx} zu weit weg nach Obstacle-Verschiebung!")
            
            print(f"   🔄 Spline wird neuberechnet")
            self._update_spline()
        
        return trajectory_adjusted


    def _reset_to_nominal_trajectory(self):
        """Fallback to the nominal trajectory if no gates are discovered or if the trajectory is too far off."""
        self._current_trajectory = self._nominal_waypoints.copy()
        self._update_spline()
        print("Fallback to nominal trajectory.")

    def _find_nearest_waypoint(self, pos):
        """Find the nearest waypoint on the current trajectory to the given position.

        This method computes the nearest point on the current trajectory to the drone's current
        position. It can be used to determine how far off the drone is from the desired path and
        to assist in trajectory adjustments.

        Args:
            pos: The current position of the drone as a numpy array [x, y, z].

        Returns:
            The nearest waypoint on the trajectory as a numpy array [x, y, z].
        """
        # Berechne Abstände zu allen Waypoints und finde den nächsten

        waypoints = self._current_trajectory
        distances = np.linalg.norm(waypoints - pos, axis=1)
        nearest_index = np.argmin(distances)
        return nearest_index


    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self.i_error[:] = 0
        self._tick = 0
        self._last_trajectory_update_tick = 0
        self._last_obstacle_update_tick = 0
        self._discovered_gates = {}
        self._discovered_obstacles = {}
        self._prev_gates_visited = np.zeros_like(self._prev_gates_visited, dtype=bool)