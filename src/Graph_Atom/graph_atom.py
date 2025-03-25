import numpy as np
import scipy as sc
from Atom.atom import Atom
from Graph_Atom.graph_sig import GraphSig
from utils.Optimizer import get_optimizer
from utils.utils import *


class GraphAtom(Atom):
    def __init__(self, optimizer_type="adam") -> None:
        super().__init__()
        self.optimizer = get_optimizer(optimizer_type)

    def set_params(self, e, R, b) -> None:
        self.e = e
        self.R = R
        self.b = b
        self.n = e.shape[0]
        self.d = R.shape[0]
        self.c = b.shape[0]

    def initialize(self, n, d, c):
        self.n = n
        self.d = d
        self.c = c

        # Initialization
        self.e = np.random.randn(self.n)
        self.e /= np.linalg.norm(self.e)

        self.R = create_Ri_symmetric_matrix_with_constrained_eigenvalues(
            self.d,
            config.rho_min,
            config.rho_max,
            noise_level=0.0,
        )

        self.b = create_multiple_noisy_probability_vectors(1, self.c, noise_level=1)[0]

        # Reset optimizer state when initializing new parameters
        self.optimizer.reset()

    def e_term(self):
        return np.outer(self.e, self.e)

    def R_term(self):
        if self.R is not None:
            return np.einsum("i,jk->ijk", ((self.e**2)), self.R)
        else:
            return None

    def b_term(self):
        return np.einsum("i,j->ij", ((self.e**2)), self.b)

    def calc_sig_term(self) -> GraphSig:
        e_term = self.e_term()
        R_term = self.R_term()
        b_term = self.b_term()
        sig = GraphSig(e_term, R_term, b_term)
        return sig

    def calc_grad(self, grad_sig: GraphSig):
        if grad_sig is None:
            return 0
        grad_ei = self.calc_grad_ei(grad_sig)
        grad_Ri = self.calc_grad_Ri(grad_sig)
        grad_bi = self.calc_grad_bi(grad_sig)

        grad = GraphAtom()
        grad.set_params(grad_ei, grad_Ri, grad_bi)
        return grad

    def calc_grad_ei(self, grad_sig: GraphSig):
        if self.R is not None:
            tmp1 = np.einsum("ijk,jk->i", grad_sig.Rv, self.R)
        else:
            tmp1 = 0

        tmp2 = np.einsum("ij,j->i", grad_sig.pi, self.b)
        tmp3 = grad_sig.L + np.diag(tmp1 + tmp2)

        grad_ei = np.einsum("ij,j->i", tmp3, self.e)

        return grad_ei

    def calc_grad_Ri(self, grad_sig: GraphSig):
        grad_Ri = np.einsum("ijk,i->jk", grad_sig.Rv, self.e**2)

        return grad_Ri

    def calc_grad_bi(self, grad_sig: GraphSig):
        grad_bi = np.einsum("i, ij-> j", self.e**2, grad_sig.pi)

        return grad_bi

    def update(self, grad, lr, use_optimizer=True):
        """Update atom parameters using optimizer or gradient descent

        Args:
            grad: Gradient object with e, R, and b attributes
            lr (float): Learning rate
            use_optimizer (bool): Whether to use optimizer or standard gradient descent
        """
        if use_optimizer:
            # Collect parameters and gradients as dictionaries for the optimizer
            params = {"e": self.e, "R": self.R, "b": self.b}

            gradients = {"e": grad.e, "R": grad.R, "b": grad.b}

            # Use optimizer to update parameters
            updated_params = self.optimizer.step(params, gradients, lr)

            # Apply updates
            self.e = updated_params["e"]
            self.R = updated_params["R"]
            self.b = updated_params["b"]
        else:
            # Original gradient descent update
            self.e -= lr * grad.e
            self.R -= lr * grad.R
            self.b -= lr * grad.b

        # Project to ensure constraints are satisfied
        self.project_to_bound()

    def project_to_bound(self):
        self.e = project_to_unit_vector(self.e)
        self.R = project_to_psd_cone(self.R, config.rho_min, config.rho_max)
        self.b = project_to_simplex(self.b, 1)

    def reset_optimizer_state(self):
        """Reset the optimizer state"""
        self.optimizer.reset()

    def get_scheduled_lr(self, lr):
        """Calculate learning rate based on scheduling strategy.

        Args:
            lr (float): Base learning rate

        Returns:
            float: Scheduled learning rate
        """
        if self.initial_lr is None:
            # Initialize scheduling parameters on first call
            self.initial_lr = lr
            self.min_lr = lr * 0.01  # Minimum LR is 1% of initial
            self.decay_rate = 0.95  # Decay by 5%
            self.decay_steps = 100  # Decay every 100 steps

        # Exponential decay schedule
        scheduled_lr = self.initial_lr * (
            self.decay_rate ** (self.t // self.decay_steps)
        )

        # Ensure learning rate doesn't go below minimum
        return max(scheduled_lr, self.min_lr)

    def __str__(self):
        return f"e: {self.e}, b: {self.b}, R: {np.round(np.linalg.eigvalsh(self.R), 2)}"

    def proximal_to_sig(self, sig: GraphSig):
        λ, evecs_large = np.linalg.eigh(sig.L)
        self.e = evecs_large[:, -1]
        λ = λ[-1] - config.mu0 / config.beta
        if λ < 0:
            λ = 0
            return λ

        if config.mu1 != 0:
            self.Ri = (np.einsum("ijk,i->jk", sig.Rv, self.e**2)) / λ
            self.Ri = project_to_psd_cone(self.Ri, config.rho_min, config.rho_max)

        if config.mu2 != 0:
            self.b = np.einsum("i, ij-> j", self.e**2, sig.pi) / λ
            self.b = project_to_simplex(self.b, 1)

        return λ

    def conditional_gradient_update(self, grad: GraphSig, itr):
        """
        Perform a conditional gradient to update the atom.

        Parameters:
        - grad: grad of function f

        Returns:
        - A new atom
        """
        max_iter = (
            config.iter_or_cycles_to_find_next_atom
            if config.mu1 != 0 or config.mu2 != 0
            else 1
        )
        th_old = np.inf
        self.initialize(self.n, self.d, self.c)

        for it in range(max_iter):
            λ = self.CG_update_step(grad, it, itr)

            th = grad.dot(self.calc_sig_term())
            if abs(th_old - th) < 1e-6:
                break
            th_old = th

        return λ

    def CG_update_step(self, grad: GraphSig, it=0, itr=0):
        """
        Perform a conditional gradient to update the atom.

        Parameters:
        - grad: grad of function f

        Returns:
        - A new atom
        """
        λ = self.update_e(grad, it, itr)
        if config.mu1 != 0:
            self.update_R(grad, it, itr)

        if config.mu2 != 0:
            self.update_b(grad)

        return λ

    ## Step 1: Fix R and b, update epsilon

    def update_e(self, grad: GraphSig, iter=0, itr=0):
        scale_factor = (config.rho_max / config.rho_min) ** 2
        if self.R is not None:
            tmp1 = np.einsum("ijk,jk->i", grad.Rv, self.R)
        else:
            tmp1 = 0

        tmp1 /= scale_factor
        tmp2 = np.einsum("ij,j->i", grad.pi, self.b)
        tmp3 = grad.L + np.diag(tmp1 + tmp2)
        N = tmp3.shape[0]
        try:
            λ, evecs_large = sc.linalg.eigh(-tmp3, subset_by_index=(N - 1, N - 1))
            self.e = evecs_large[:, 0]
        except:
            λ, evecs_large = np.linalg.eigh(-tmp3)
            self.e = evecs_large[:, -1]
            λ = λ[-1]

        if plot:
            sub_folder = f"{save_path}{itr}/e/"
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)
            plt.figure()
            plt.plot(self.e, "-*")
            plt.savefig(f"{sub_folder}{iter}.png")
            plt.close()

        return λ

    ## Step 2: Fix epsilon and b, update R
    def update_R(self, grad: GraphSig, iter=0, itr=0):
        grad_R = self.calc_grad_Ri(grad)

        S, U = np.linalg.eigh(grad_R)

        if plot:
            sub_folder = f"{save_path}{itr}/R/"
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)

            plt.figure()
            plt.bar(np.arange(S.shape[0]), S)
            plt.savefig(f"{sub_folder}{iter}.png")
            plt.close()

        mask1 = S < 0
        S[mask1] = config.rho_max
        S[~mask1] = config.rho_min
        self.R = U @ np.diag(S) @ U.T

    ## Step 3: Fix epsilon and R, update b (closed-form solution)
    def update_b(self, grad: GraphSig):
        a = self.calc_grad_bi(grad)
        self.b = np.zeros(grad.pi.shape[-1], dtype=float)
        self.b[np.argmin(a)] = 1
