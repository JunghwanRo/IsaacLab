import torch


class RotorDynamics:
    def __init__(self, num_envs, device, Td_rotor=None, act_dt=None):
        self.num_envs = num_envs
        self.device = device
        self.Td_rotor = Td_rotor
        self.act_dt = act_dt
        self.thrusts = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.torques = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.prev_action = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def smooth_action(self, action: torch.Tensor) -> torch.Tensor:
        """Smooth the action based on the time constant and timestep."""
        if self.Td_rotor is not None and self.act_dt is not None and self.Td_rotor > 0.0:
            alpha = self.act_dt / self.Td_rotor
            smoothed_action = self.prev_action + alpha * (action - self.prev_action)
            self.prev_action = smoothed_action
            return smoothed_action
        return action

    def action_to_thrust(self, action: torch.Tensor) -> torch.Tensor:
        """Convert actions to thrust."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_thrust_and_torque(self, action: torch.Tensor, ccw: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate thrust and torque from action."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class SimpleRotor(RotorDynamics):
    def __init__(
        self,
        num_envs: int,
        num_rotors: int,
        is_ccw: bool,
        k_tau: float,
        k_tau_neg: float,
        min_thrust: float,
        max_thrust: float,
        Td_rotor: float,
        act_dt: float,
        device: torch.device,
    ):
        super().__init__(num_envs, device, Td_rotor, act_dt)
        self.num_rotors = num_rotors
        self.is_ccw = is_ccw
        self.k_tau = k_tau
        self.k_tau_neg = k_tau_neg
        self.min_thrust = 0.0
        self.max_thrust = max_thrust

    def action_to_thrust(self, action: torch.Tensor) -> torch.Tensor:
        """Convert action to thrust. Action range is [-1, 1]."""
        # Map action from [-1, 1] to [0, 1]
        normalized_action = (action + 1.0) / 2.0
        self.thrusts = normalized_action * self.max_thrust
        return self.thrusts

    def thrust_to_torque(self, thrust: torch.Tensor) -> torch.Tensor:
        """Convert thrust to torque."""
        self.torques = self.k_tau * thrust * (-1 if self.is_ccw else 1)
        return self.torques

    def get_thrust_and_torque(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate thrust and torque from action."""
        smoothed_action = self.smooth_action(action)
        self.thrusts = self.action_to_thrust(smoothed_action)
        self.torques = self.thrust_to_torque(self.thrusts)
        return self.thrusts, self.torques


class SimpleReverseRotor(RotorDynamics):
    def __init__(
        self,
        num_envs: int,
        num_rotors: int,
        is_ccw: bool,
        k_tau: float,
        k_tau_neg: float,
        min_thrust: float,
        max_thrust: float,
        Td_rotor: float,
        act_dt: float,
        device: torch.device,
    ):
        super().__init__(num_envs, device, Td_rotor, act_dt)
        self.num_rotors = num_rotors
        self.is_ccw = is_ccw
        self.k_tau = k_tau
        self.k_tau_neg = k_tau_neg
        self.min_thrust = min_thrust
        self.max_thrust = max_thrust

    def action_to_thrust(self, action: torch.Tensor) -> torch.Tensor:
        """Convert action to thrust. Action range is [-1, 1]."""
        thrust = torch.where(action > 0, action * self.max_thrust, -action * self.min_thrust)
        self.thrusts = thrust
        return self.thrusts

    def thrust_to_torque(self, thrust: torch.Tensor) -> torch.Tensor:
        """Convert thrust to torque."""
        k_tau = torch.where(thrust > 0, self.k_tau, self.k_tau_neg)
        self.torques = k_tau * thrust * (-1 if self.is_ccw else 1)
        return self.torques

    def get_thrust_and_torque(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate thrust and torque from action."""
        smoothed_action = self.smooth_action(action)
        self.thrusts = self.action_to_thrust(smoothed_action)
        self.torques = self.thrust_to_torque(self.thrusts)
        return self.thrusts, self.torques


class LinearIntRotor(RotorDynamics):
    """Rotor dynamics with linear interpolation between points.
    Do not need it unless the rotor has very specific feature and dynamics relation is not linear."""

    pass


def create_rotor(rotor_type: str, num_envs: int, num_rotors: int, device: torch.device, **kwargs) -> RotorDynamics:
    if rotor_type == "simple":
        return SimpleRotor(num_envs, num_rotors, **kwargs, device=device)
    elif rotor_type == "simplereverse":
        return SimpleReverseRotor(num_envs, num_rotors, **kwargs, device=device)
    elif rotor_type == "linear":
        return LinearIntRotor(num_envs, num_rotors, **kwargs, device=device)
    else:
        raise ValueError(f"Unknown rotor type: {rotor_type}")
