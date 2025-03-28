import torch
from dataclasses import MISSING
from isaaclab.utils import configclass


@configclass
class AddedMassCfg:
    """Configuration for added-mass effects.

    The added mass may be provided as a list of six values (for a diagonal matrix)
    or as a 6x6 nested list. The parameter `alpha` controls the low-pass filtering
    of the acceleration estimate, and `sim_dt` is the simulation timestep.
    """

    added_mass: list = MISSING  # e.g. [m_u, m_v, m_w, m_p, m_q, m_r] for a diagonal matrix
    alpha: float = 0.5  # Low-pass filter coefficient
    sim_dt: float = 0.01  # Simulation timestep (s)
    apply_if_z_below: float = 0.0


class AddedMass:
    def __init__(self, num_envs: int, device: str, sim_dt: float, cfg: AddedMassCfg):
        self.num_envs = num_envs
        self.device = device
        self.sim_dt = sim_dt
        self.cfg = cfg

        # Convert the added_mass parameter into a tensor and, if given as a 1D list,
        # create a diagonal matrix.
        self.added_mass_matrix = torch.tensor(cfg.added_mass, device=device, dtype=torch.float32)
        if self.added_mass_matrix.ndim == 1:
            self.added_mass_matrix = torch.diag(self.added_mass_matrix)
        # Use the same added mass matrix for all environments.
        self.alpha = cfg.alpha

        # Initialize buffers for the previous velocity and the filtered acceleration.
        self.prev_velocity = torch.zeros((num_envs, 6), device=device, dtype=torch.float32)
        self.filtered_acc = torch.zeros((num_envs, 6), device=device, dtype=torch.float32)

    def compute_added_mass_force(self, current_velocity: torch.Tensor, positions_z: torch.Tensor) -> torch.Tensor:
        """
        Compute the added-mass force using a low-pass filtered acceleration estimate.

        Args:
            current_velocity (torch.Tensor): Current velocity (6 DOF) of the vehicle in the body frame. Shape: (num_envs, 6).

        Returns:
            torch.Tensor: The added-mass force (per environment), shape (num_envs, 6).
        """
        # Estimate acceleration (finite difference approximation).
        acceleration_est = (current_velocity - self.prev_velocity) / self.sim_dt

        # If any element of the acceleration estimate exceeds 30 m/s² in absolute value, set it to 0.
        # TODO: debug this why sometimes the acceleration is too high. (mainly guess the reset)
        acceleration_est = torch.where(torch.abs(acceleration_est) > 30.0, torch.zeros_like(acceleration_est), acceleration_est)

        # Apply low-pass filtering to reduce instability.
        self.filtered_acc = self.alpha * acceleration_est + (1 - self.alpha) * self.filtered_acc

        # Compute the added mass force: F_added = - M_A * filtered_acc
        # (Matrix multiplication: (6x6) @ (6, 1) for each environment.)
        force = -torch.matmul(self.added_mass_matrix, self.filtered_acc.unsqueeze(-1)).squeeze(-1)

        # Update the stored previous velocity.
        self.prev_velocity = current_velocity.clone()

        # Only apply the added-mass force if underwater (positions_z < apply_if_z_below).
        mask = (positions_z < self.cfg.apply_if_z_below).float().unsqueeze(-1)  # Shape: (num_envs, 1)
        force = force * mask

        # Debug
        # force = torch.zeros_like(force)

        return force
