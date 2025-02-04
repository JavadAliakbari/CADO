import numpy as np
import scipy as sc
from Atom.atom import Atom

# from Atom.coefs import Coefs
from Graph_Atom.graph_sig import GraphSig
from utils.utils import *


class GraphAtom(Atom):
    def __init__(self) -> None:
        super().__init__()

    def set_params(self, e, R, b) -> None:
        self.e = e
        self.R = R
        self.b = b
        # self.coef = coef
        self.n = e.shape[0]
        self.d = R.shape[0]
        self.c = b.shape[0]

    def initialize(self, n, d, c):
        self.n = n
        self.d = d
        self.c = c
        # Initialization
        # self.e = np.zeros(self.n)
        # self.e[:4] = 0.5
        self.e = np.random.randn(self.n)
        # self.e = 0.5 + 0.5 * np.random.randn(self.n)
        # self.e = np.ones(self.n)
        self.e /= np.linalg.norm(self.e)
        self.R = create_Ri_symmetric_matrix_with_constrained_eigenvalues(
            self.d,
            0,
            config.rho_max - config.rho_min,
            noise_level=0.0,
        )
        self.b = create_multiple_noisy_probability_vectors(1, self.c, noise_level=1)[0]

        # self.R = np.zeros((self.d, self.d))
        # self.b = np.zeros(self.c)

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
        sig = GraphSig(e_term, R_term, b_term, add_eps=True)
        return sig

    # def update_e(self, e):
    #     self.e = e

    # def update_R(self, R):
    #     self.R = R

    # def update_b(self, b):
    #     self.b = b

    def calc_grad(self, grad_sig: GraphSig):
        if grad_sig is None:
            return 0
        grad_ei = self.calc_grad_ei(grad_sig) / grad_sig.n
        grad_Ri = self.calc_grad_Ri(grad_sig) / grad_sig.d
        grad_bi = self.calc_grad_bi(grad_sig) / grad_sig.c

        grad = GraphAtom()
        grad.set_params(grad_ei, grad_Ri, grad_bi)
        return grad

    def calc_grad_ei(self, grad_sig: GraphSig):
        scale_factor = (config.rho_max / config.rho_min) ** 2
        if self.R is not None:
            tmp1 = np.einsum("ijk,jk->i", grad_sig.Rv, self.R)
        else:
            tmp1 = 0

        tmp1 /= scale_factor
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

    def update(self, grad, lr):
        self.e -= lr * grad.e
        self.R -= lr * grad.R
        self.b -= lr * grad.b

        self.project_to_bound()

    def project_to_bound(self):
        self.e = project_to_unit_vector(self.e)
        self.R = project_to_psd_cone(self.R, config.rho_min, config.rho_max)
        self.b = project_to_simplex(self.b, 1)

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

        for it in range(max_iter):
            λ = self.update_e(grad, it, itr)

            if config.mu1 != 0:
                self.update_R(grad, it, itr)

            if config.mu2 != 0:
                self.update_b(grad)

            th = grad.dot(self.calc_sig_term())
            if abs(th_old - th) < 1e-6:
                break
            th_old = th

        return λ

    ## Step 1: Fix R and b, update epsilon

    def update_e(self, grad: GraphSig, iter=0, itr=0):
        scale_factor = (config.rho_max / config.rho_min) ** 2
        if self.R is not None:
            tmp1 = np.einsum("ijk,jk->i", grad.Rv, self.R)
        else:
            tmp1 = 0

        tmp1 /= scale_factor
        # tmp1 /= grad.n
        tmp2 = np.einsum("ij,j->i", grad.pi, self.b)
        tmp3 = grad.L + np.diag(tmp1 + tmp2)
        # S, eigvecs = np.linalg.eigh(-tmp3)
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
    def update_R(self, grad: GraphSig, iter, itr=0):
        grad_R = self.calc_grad_Ri(grad)
        # grad_R = np.einsum("ijk,i->jk", grad.Rv, atom.e**2)

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
        S[mask1] = config.rho_max - config.rho_min
        S[~mask1] = 0
        self.R = U @ np.diag(S) @ U.T

    ## Step 3: Fix epsilon and R, update b (closed-form solution)
    def update_b(self, grad: GraphSig):
        a = self.calc_grad_bi(grad)
        # a = np.einsum("i, ij-> j", atom.e**2, grad.pi)
        self.b = np.zeros(grad.pi.shape[-1], dtype=float)
        self.b[np.argmin(a)] = 1
