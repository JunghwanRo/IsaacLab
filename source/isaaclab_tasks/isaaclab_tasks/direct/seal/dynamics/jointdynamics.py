import torch


class JointDynamics:
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.joint_efforts = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def state_to_effort(self, joint_states: torch.Tensor) -> torch.Tensor:
        """Convert joint states to efforts."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_joint_effort(self, joint_states: torch.Tensor) -> torch.Tensor:
        """Calculate joint efforts from joint states."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class LinearIntJoint(JointDynamics):
    def __init__(
        self,
        num_envs: int,
        num_int_points: int,
        pos_to_effort_graph: torch.Tensor,
        device: torch.device,
    ):
        super().__init__(num_envs, device)
        self.num_int_points = num_int_points
        self.pos_to_effort_graph = pos_to_effort_graph.to(device)

        # Precompute sorted indices and slopes
        self.precompute_interpolation()

    def precompute_interpolation(self):
        """Precompute sorted indices and slopes for interpolation."""
        # Sort x values and associated y values
        x_vals = self.pos_to_effort_graph[:, 0]
        y_vals = self.pos_to_effort_graph[:, 1]

        sorted_indices = torch.argsort(x_vals)
        self.x_vals = x_vals[sorted_indices].contiguous()
        self.y_vals = y_vals[sorted_indices].contiguous()

        # Calculate slopes for interpolation
        x_diff = self.x_vals[1:] - self.x_vals[:-1]
        y_diff = self.y_vals[1:] - self.y_vals[:-1]
        self.slopes = y_diff / x_diff

    def linear_interpolate(self, x_query):
        """Perform linear interpolation using precomputed slopes and indices."""
        # Ensure x_query is contiguous
        x_query = x_query.contiguous().unsqueeze(1)

        # Find the indices for the lower and upper bounds
        x_below = torch.searchsorted(self.x_vals, x_query, right=True) - 1
        x_above = x_below + 1

        # Clamp indices to ensure they are within the valid range
        x_below = torch.clamp(x_below, 0, len(self.x_vals) - 2)
        x_above = torch.clamp(x_above, 1, len(self.x_vals) - 1)

        # Get the x and y values for the lower bounds of the interval
        x0 = self.x_vals[x_below]
        y0 = self.y_vals[x_below]
        slope = self.slopes[x_below]

        # Perform the interpolation
        y_query = y0 + slope * (x_query - x0)

        return y_query.squeeze()

    def state_to_effort(self, joint_states: torch.Tensor) -> torch.Tensor:
        """Convert joint positions to efforts using linear interpolation."""
        return self.linear_interpolate(joint_states)

    def get_joint_effort(self, joint_states: torch.Tensor) -> torch.Tensor:
        """Calculate joint efforts from joint states."""
        self.joint_efforts = self.state_to_effort(joint_states)
        return self.joint_efforts


def create_joint(joint_type: str, num_envs: int, device: torch.device, **kwargs) -> JointDynamics:
    if joint_type == "linearint":
        return LinearIntJoint(num_envs, **kwargs, device=device)
    else:
        raise ValueError(f"Unknown joint type: {joint_type}")
