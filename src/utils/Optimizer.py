import numpy as np
from abc import ABC, abstractmethod
from utils.utils import optimizer  # Import config from utils


class Optimizer(ABC):
    """Abstract base class for optimizers"""

    @abstractmethod
    def step(self, params, gradients):
        """Update parameters based on gradients"""
        pass

    @abstractmethod
    def reset(self):
        """Reset optimizer state"""
        pass


class Adam(Optimizer):
    """Adam optimizer with learning rate scheduling"""

    def __init__(self, beta1=None, beta2=None, epsilon=None):
        """Initialize Adam optimizer

        Args:
            beta1 (float, optional): Exponential decay rate for first moment estimates
            beta2 (float, optional): Exponential decay rate for second moment estimates
            epsilon (float, optional): Small constant to prevent division by zero
        """
        # Use values from config or fallback to defaults
        self.beta1 = beta1 if beta1 is not None else optimizer.beta1
        self.beta2 = beta2 if beta2 is not None else optimizer.beta2
        self.epsilon = epsilon if epsilon is not None else optimizer.epsilon

        # Optimizer state
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0  # Time step

        # Learning rate scheduling parameters from config
        self.initial_lr = optimizer.lr
        self.min_lr = optimizer.min
        self.decay_rate = optimizer.decay_rate
        self.decay_steps = optimizer.decay_steps

    def step(self, params, gradients):
        """Perform one optimization step

        Args:
            params (dict): Dictionary of parameters to update
            gradients (dict): Dictionary of parameter gradients
            lr (float): Base learning rate

        Returns:
            dict: Updated parameters
        """
        # Increment time step
        self.t += 1

        # Get scheduled learning rate
        scheduled_lr = self._get_scheduled_lr()

        # Update for each parameter
        for key in params:
            if key not in gradients:
                continue

            # Initialize momentum variables if needed
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]

            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (
                gradients[key] ** 2
            )

            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1**self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            # Update parameter
            params[key] -= scheduled_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params

    def reset(self):
        """Reset optimizer state"""
        self.m = {}
        self.v = {}
        self.t = 0
        self.initial_lr = None
        self.min_lr = None
        self.decay_rate = None
        self.decay_steps = None

    def _get_scheduled_lr(self):
        """Calculate learning rate based on scheduling strategy

        Args:
            lr (float): Base learning rate

        Returns:
            float: Scheduled learning rate
        """
        if self.initial_lr is None:
            # Use values from config if available, otherwise set defaults
            self.initial_lr = optimizer.lr
            self.min_lr = optimizer.min
            self.decay_rate = optimizer.decay_rate
            self.decay_steps = optimizer.decay_steps

        # Exponential decay schedule
        scheduled_lr = self.initial_lr * (
            self.decay_rate ** (self.t // self.decay_steps)
        )

        # Ensure learning rate doesn't go below minimum
        return max(scheduled_lr, self.min_lr)

    def set_schedule_params(
        self, initial_lr=None, min_lr=None, decay_rate=None, decay_steps=None
    ):
        """Set learning rate schedule parameters

        Args:
            initial_lr (float, optional): Initial learning rate
            min_lr (float, optional): Minimum learning rate
            decay_rate (float, optional): Rate of decay
            decay_steps (int, optional): Steps between decay
        """
        if initial_lr is not None:
            self.initial_lr = initial_lr
        if min_lr is not None:
            self.min_lr = min_lr
        if decay_rate is not None:
            self.decay_rate = decay_rate
        if decay_steps is not None:
            self.decay_steps = decay_steps


# You can add other optimizers here like SGD, RMSProp, etc.
class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with learning rate scheduling"""

    def __init__(self, momentum=None):
        """Initialize SGD optimizer

        Args:
            momentum (float, optional): Momentum factor
        """
        # Use values from config or fallback to defaults
        self.momentum = momentum if momentum is not None else optimizer.momentum

        self.velocity = {}  # Velocity for momentum
        self.t = 0  # Time step

        # Learning rate scheduling parameters from config
        self.initial_lr = optimizer.lr
        self.min_lr = optimizer.min
        self.decay_rate = optimizer.decay_rate
        self.decay_steps = optimizer.decay_steps

    def step(self, params, gradients):
        """Perform one optimization step

        Args:
            params (dict): Dictionary of parameters to update
            gradients (dict): Dictionary of parameter gradients
            lr (float): Base learning rate

        Returns:
            dict: Updated parameters
        """
        # Increment time step
        self.t += 1

        # Get scheduled learning rate
        scheduled_lr = self._get_scheduled_lr()

        # Update for each parameter
        for key in params:
            if key not in gradients:
                continue

            # Initialize velocity if needed
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])

            # Update velocity
            self.velocity[key] = (
                self.momentum * self.velocity[key] + scheduled_lr * gradients[key]
            )

            # Update parameter
            params[key] -= self.velocity[key]

        return params

    def reset(self):
        """Reset optimizer state"""
        self.velocity = {}
        self.t = 0
        self.initial_lr = None
        self.min_lr = None
        self.decay_rate = None
        self.decay_steps = None

    def _get_scheduled_lr(self):
        """Calculate learning rate based on scheduling strategy

        Args:
            lr (float): Base learning rate

        Returns:
            float: Scheduled learning rate
        """
        if self.initial_lr is None:
            # Use values from config if available, otherwise set defaults
            self.initial_lr = optimizer.lr
            self.min_lr = optimizer.min
            self.decay_rate = optimizer.decay_rate
            self.decay_steps = optimizer.decay_steps

        # Exponential decay schedule
        scheduled_lr = self.initial_lr * (
            self.decay_rate ** (self.t // self.decay_steps)
        )

        # Ensure learning rate doesn't go below minimum
        return max(scheduled_lr, self.min_lr)


def get_optimizer(optimizer_type=None, **kwargs):
    """Factory function to create optimizers

    Args:
        optimizer_type (str, optional): Type of optimizer (adam, sgd)
        **kwargs: Additional parameters for the optimizer

    Returns:
        Optimizer: The requested optimizer instance
    """
    # Use optimizer type from config if not specified
    if optimizer_type is None:
        optimizer_type = optimizer.type

    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adam":
        return Adam(**kwargs)
    elif optimizer_type == "sgd":
        return SGD(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
