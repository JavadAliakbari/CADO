import numpy as np
import scipy as sc
from Atom.atom import Atom
from Graph_Atom.graph_sig import GraphSig
from utils.SBM import SparseAbar
from utils.Optimizer import get_optimizer
from utils.utils import *


# Add a lazy outer product helper for rank-1 matrices
class OuterProduct:
    def __init__(self, vector, coef=1):
        self.vector = vector
        self.coef = coef

    @property
    def shape(self):
        n = self.vector.shape[0]
        return (n, n)

    def __array__(self, dtype=None):
        return self.coef * np.outer(self.vector, self.vector)

    # Added: efficient lazy addition functionality
    def __add__(self, other):
        if isinstance(other, OuterProduct):
            return LazyOuterSum([self, other])
        elif isinstance(other, LazyOuterSum):
            return LazyOuterSum([self] + other.terms)
        elif isinstance(other, (int, float)):
            if other == 0:
                return self
            else:
                return self.__array__() + other
        else:
            return self.__array__() + other

    def __sub__(self, other):
        if isinstance(other, OuterProduct):
            return LazyOuterSum([self, -1 * other])
        elif isinstance(other, LazyOuterSum):
            return LazyOuterSum([self] + (-1 * other).terms)
        elif isinstance(other, (int, float)):
            if other == 0:
                return self
            else:
                return self.__array__() + other
        else:
            return self.__array__() + other

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, c):
        if isinstance(c, (int, float)):
            return OuterProduct(self.vector, self.coef * c)
        elif isinstance(c, (np.ndarray, SparseAbar)):
            # Faster implementation using two dot products instead of np.einsum
            dot_product = c @ self.vector
            return self.coef * (self.vector @ dot_product)

    def __rmul__(self, c):
        return self * c


# New class to lazily sum multiple OuterProduct objects
class LazyOuterSum:
    def __init__(self, terms):
        self.terms = terms  # list of OuterProduct instances

    @property
    def shape(self):
        n = self.terms[0].vector.shape[0]
        return (n, n)

    def __array__(self, dtype=None):
        total = 0
        for term in self.terms:
            total += np.array(term)
        return total

    def __add__(self, other):
        if isinstance(other, OuterProduct):
            return LazyOuterSum(self.terms + [other])
        elif isinstance(other, LazyOuterSum):
            return LazyOuterSum(self.terms + other.terms)
        else:
            return np.array(self) + other

    def __sub__(self, other):
        if isinstance(other, OuterProduct):
            return LazyOuterSum(self.terms + [-1 * other])
        elif isinstance(other, LazyOuterSum):
            return LazyOuterSum(self.terms + [-1 * term for term in other.terms])
        else:
            return np.array(self) + other

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, c):
        if isinstance(c, (int, float)):
            return LazyOuterSum([term * c for term in self.terms])
        elif isinstance(c, (np.ndarray, SparseAbar)):
            return sum([term * c for term in self.terms])

    def __rmul__(self, c):
        return self * c


class GraphAtom(Atom):
    def __init__(self) -> None:
        super().__init__()
        self.optimizer = get_optimizer(config.optimizer.type)

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
        # Changed: return a lazy outer product instead of constructing the full matrix immediately
        return OuterProduct(self.e)

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
        grad_Ri = self.calc_grad_Ri(grad_sig.Rv)
        grad_bi = self.calc_grad_bi(grad_sig.pi)

        grad = GraphAtom()
        grad.set_params(grad_ei, grad_Ri, grad_bi)
        return grad

    def calc_grad_ei(self, grad_sig: GraphSig):
        if self.R is not None and grad_sig.Rv is not None:
            tmp1 = np.einsum("ijk,jk->i", grad_sig.Rv, self.R)
        else:
            tmp1 = 0

        tmp2 = np.einsum("ij,j->i", grad_sig.pi, self.b)
        # tmp3 = grad_sig.L + np.diag(tmp1 + tmp2)

        grad_ei_1 = (tmp1 + tmp2) * self.e
        grad_ei_2 = grad_sig.L @ self.e
        # grad_ei_2 = np.einsum("ij, j-> i", grad_sig.L, self.e)
        grad_ei = grad_ei_1 + grad_ei_2

        return grad_ei

    def calc_grad_Ri(self, grad_Rv):
        if grad_Rv is not None:
            grad_Ri = np.einsum("ijk,i->jk", grad_Rv, self.e**2)

            return grad_Ri
        else:
            return None

    def calc_grad_bi(self, grad_pi):
        grad_bi = np.einsum("i, ij-> j", self.e**2, grad_pi)

        return grad_bi

    def update(self, grad):
        """Update atom parameters using optimizer or gradient descent

        Args:
            grad: Gradient object with e, R, and b attributes
            lr (float): Learning rate
            use_optimizer (bool): Whether to use optimizer or standard gradient descent
        """
        # Collect parameters and gradients as dictionaries for the optimizer
        params = {"e": self.e, "R": self.R, "b": self.b}

        gradients = {"e": grad.e, "R": grad.R, "b": grad.b}

        # Use optimizer to update parameters
        updated_params = self.optimizer.step(params, gradients)

        # Apply updates
        self.e = updated_params["e"]
        self.R = updated_params["R"]
        self.b = updated_params["b"]

        # Project to ensure constraints are satisfied
        self.project_to_bound()

    def project_to_bound(self):
        self.e = project_to_unit_vector(self.e)
        self.R = project_to_psd_cone(self.R, config.rho_min, config.rho_max)
        self.b = project_to_simplex(self.b, 1)

    def reset_optimizer_state(self):
        """Reset the optimizer state"""
        self.optimizer.reset()

    def __str__(self):
        return f"e: {self.e}, b: {self.b}, R: {np.round(np.linalg.eigvalsh(self.R), 2)}"

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
    def update_R(self, grad_Rv, iter=0, itr=0):
        if grad_Rv is None:
            return
        grad_R = self.calc_grad_Ri(grad_Rv)

        S, U = np.linalg.eigh(grad_R)

        # if plot:
        #     sub_folder = f"{save_path}{itr}/R/"
        #     if not os.path.exists(sub_folder):
        #         os.makedirs(sub_folder)

        #     plt.figure()
        #     plt.bar(np.arange(S.shape[0]), S)
        #     plt.savefig(f"{sub_folder}{iter}.png")
        #     plt.close()

        mask1 = S < 0
        S[mask1] = config.rho_max
        S[~mask1] = config.rho_min
        self.R = np.einsum("i,ji,ki->jk", S, U, U)

        # S[mask1] = config.rho_max -  config.rho_min
        # S[~mask1] = 0
        # self.R = U @ np.diag(S) @ U.T + config.rho_min * np.eye(self.R.shape[0])

    ## Step 3: Fix epsilon and R, update b (closed-form solution)
    def update_b(self, grad_pi):
        a = self.calc_grad_bi(grad_pi)
        self.b = np.zeros(grad_pi.shape[-1], dtype=float)
        self.b[np.argmin(a)] = 1

    def __add__(self, atom_):
        e = np.sqrt(self.e**2 + atom_.e**2)
        R = self.R + atom_.R
        b = self.b + atom_.b
        # R = project_to_psd_cone(R, config.rho_min, config.rho_max)

        atom = GraphAtom()
        atom.set_params(e, R, b)
        return atom

    def __mul__(self, c):
        e = np.sqrt(c) * self.e
        R = c * self.R
        b = c * self.b
        atom = GraphAtom()
        atom.set_params(e, R, b)
        return atom

    def __rmul__(self, c):
        e = np.sqrt(c) * self.e
        R = c * self.R
        b = c * self.b
        atom = GraphAtom()
        atom.set_params(e, R, b)
        return atom
