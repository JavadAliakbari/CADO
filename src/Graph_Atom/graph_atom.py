import numpy as np
import scipy as sc
from Atom.atom import Atom
from Graph_Atom.graph_sig import GraphSig
from utils.outer_product import OuterProduct
from utils.Optimizer import get_optimizer
from utils.utils import *


class GraphAtom(Atom):
    def __init__(self) -> None:
        super().__init__()
        self.optimizer = get_optimizer(optimizer.type)
        self.r = None

    def set_params(self, e, theta, b) -> None:
        self.e = e
        self.theta = theta
        self.b = b
        self.n = e.shape[0]
        self.d = theta.shape[0]
        self.c = b.shape[0]

    def initialize(self, n, d, c):
        self.n = n
        self.d = d
        self.c = c

        # Initialization
        self.e = np.random.randn(self.n)
        self.e /= np.linalg.norm(self.e)

        # Initialize theta as a random d-dimensional vector with norm <= tau
        self.theta = np.random.randn(self.d)
        self.theta = project_to_ball(self.theta, data.tau)

        self.b = create_multiple_noisy_probability_vectors(1, self.c, noise_level=1)[0]

        # Reset optimizer state when initializing new parameters
        self.optimizer.reset()

    def e_term(self):
        # Changed: return a lazy outer product instead of constructing the full matrix immediately
        return OuterProduct(self.e)

    def R_term(self):
        """Returns cluster centers for each node: u_v = e_i^2 * theta"""
        if self.theta is not None:
            # Return (n x d) matrix of cluster centers, one for each node
            return np.einsum("i,j->ij", (self.e**2), self.theta)
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
        grad_theta_i = self.calc_grad_theta_i(grad_sig.u_v)
        grad_bi = self.calc_grad_bi(grad_sig.pi)

        grad = GraphAtom()
        grad.set_params(grad_ei, grad_theta_i, grad_bi)
        return grad

    def calc_grad_ei(self, grad_sig: GraphSig):
        if self.theta is not None and grad_sig.u_v is not None:
            # grad_u_v is (n x d), theta is (d,)
            tmp1 = np.einsum("ij,j->i", grad_sig.u_v, self.theta)
        else:
            tmp1 = 0

        tmp2 = np.einsum("ij,j->i", grad_sig.pi, self.b)
        # tmp3 = grad_sig.L + np.diag(tmp1 + tmp2)

        grad_ei_1 = (tmp1 + tmp2) * self.e
        grad_ei_2 = grad_sig.L @ self.e
        # grad_ei_2 = np.einsum("ij, j-> i", grad_sig.L, self.e)
        grad_ei = grad_ei_1 + grad_ei_2

        if plot:
            l = [tmp1, tmp2, self.e, grad_ei_1, grad_ei_2, grad_ei, tmp1 * self.e]
            for i, ll in enumerate(l):
                plt.figure()
                plt.plot(ll)
                plt.savefig(f"{save_path}x{i}.png")
        return grad_ei

    def calc_grad_theta_i(self, grad_u_v):
        """Compute gradient w.r.t. theta: grad_theta = sum_v (e_v^2 * grad_u_v)"""
        if grad_u_v is not None:
            # grad_u_v is (n x d), e^2 is (n,)
            grad_theta = np.einsum("ij,i->j", grad_u_v, self.e**2)
            return grad_theta
        else:
            return None

    def calc_grad_bi(self, grad_pi):
        grad_bi = np.einsum("i, ij-> j", self.e**2, grad_pi)

        return grad_bi

    def update(self, grad):
        """Update atom parameters using optimizer or gradient descent

        Args:
            grad: Gradient object with e, theta, and b attributes
            lr (float): Learning rate
            use_optimizer (bool): Whether to use optimizer or standard gradient descent
        """
        # Collect parameters and gradients as dictionaries for the optimizer
        params = {"e": self.e, "theta": self.theta, "b": self.b}

        gradients = {"e": grad.e, "theta": grad.theta, "b": grad.b}

        # Use optimizer to update parameters
        updated_params = self.optimizer.step(params, gradients)

        # Apply updates
        self.e = updated_params["e"]
        self.theta = updated_params["theta"]
        self.b = updated_params["b"]

        # Project to ensure constraints are satisfied
        self.project_to_bound()

    def project_to_bound(self):
        self.e = project_to_unit_vector(self.e)
        # Project theta to have norm <= tau
        self.theta = project_to_ball(self.theta, data.tau)
        self.b = project_to_simplex(self.b, 1)

    def reset_optimizer_state(self):
        """Reset the optimizer state"""
        self.optimizer.reset()

    def __str__(self):
        return f"e: {self.e}, b: {self.b}, theta: {np.round(self.theta, 2)}, ||theta||: {np.linalg.norm(self.theta):.2f}"

    ## Step 1: Fix theta and b, update epsilon
    def update_e(self, grad: GraphSig, iter=0, itr=0):
        if self.theta is not None:
            # grad.u_v is (n x d), theta is (d,)
            tmp1 = np.einsum("ij,j->i", grad.u_v, self.theta)
        else:
            tmp1 = 0

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

    ## Step 2: Fix epsilon and b, update theta
    def update_theta(self, grad_u_v):
        """Update theta to minimize objective with constraint ||theta|| <= tau"""
        if grad_u_v is None:
            return
        grad_theta = self.calc_grad_theta_i(grad_u_v)

        # For constrained optimization: theta = -tau * grad_theta / ||grad_theta||
        # if the unconstrained solution violates the constraint
        grad_norm = np.linalg.norm(grad_theta)
        if grad_norm > 0:
            # Optimal solution: theta in direction of -grad_theta
            self.theta = -data.tau * grad_theta / grad_norm
        else:
            # If gradient is zero, keep current theta
            pass

    ## Step 3: Fix epsilon and R, update b (closed-form solution)
    def update_b(self, grad_pi):
        a = self.calc_grad_bi(grad_pi)
        self.b = np.zeros(grad_pi.shape[-1], dtype=float)
        self.b[np.argmin(a)] = 1

    def __add__(self, atom_):
        e = np.sqrt(self.e**2 + atom_.e**2)
        theta = self.theta + atom_.theta
        b = self.b + atom_.b

        atom = GraphAtom()
        atom.set_params(e, theta, b)
        return atom

    def __mul__(self, c):
        e = np.sqrt(c) * self.e
        theta = c * self.theta
        b = c * self.b
        atom = GraphAtom()
        atom.set_params(e, theta, b)
        return atom

    def __rmul__(self, c):
        e = np.sqrt(c) * self.e
        theta = c * self.theta
        b = c * self.b
        atom = GraphAtom()
        atom.set_params(e, theta, b)
        return atom
