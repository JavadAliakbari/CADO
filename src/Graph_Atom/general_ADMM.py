from Atom.sig import Sig
from utils.utils import *
from utils.algorithm_utils_orig import *


class ADMM_sig:
    def __init__(self, n, d, c, graph=None):
        if config.warm_start == "random":
            self.atoms, self.coefs = create_random_atoms(n, d, c, c + 3)
        elif config.warm_start == "optimal":
            self.atoms, self.coefs = create_optimal_atoms(graph)
        self.num_atoms = len(self.atoms)
        self.Z = EmptySig(n, d, c)
        self.graph = graph

    def get_Lp(self):
        return self.atoms.calc_sig(self.coefs)

    def admm_value(self, sig: Sig):
        admm_sig = self.get_Lp()
        d = sig - admm_sig
        primal_term = Sig.dot(d, d)
        lambda_term = Sig.dot(self.Z, d)
        return lambda_term + config.beta / 2 * primal_term

    def ADMM_update(self, sig: Sig, f: GraphFunc):
        admm_sig = self.get_Lp()
        d = sig - admm_sig
        self.Z += config.beta * d

        if config.Abar:
            sig = admm_sig - (f.calc_grad(sig) + self.Z) / config.beta
        else:
            ss = EmptySig(
                self.graph.num_nodes, self.graph.num_features, self.graph.num_clusters
            )
            ss.L = self.graph.A
            sig = config.beta / (config.beta + 1) * admm_sig + 1 / (config.beta + 1) * (
                ss - self.Z
            )
        sig.project()

        # max Lambda

        M = d + self.Z / config.beta
        for ind in range(self.num_atoms):
            M += self.atoms[ind].calc_sig_term() * self.coefs[ind]
            λ = self.atoms[ind].proximal_to_sig(M)
            self.coefs[ind] = λ
            # c_new = project_to_simplex(self.coefs, sig.n)
            # self.coefs.update_coefficients(c_new)
            M -= self.atoms[ind].calc_sig_term() * λ

        return sig, self.coefs, self.atoms

    def ADMM_metric(self, sig: Sig):
        admm_sig = self.get_Lp()
        d = sig - admm_sig
        diff = d.dot(d)
        lambda_term = d.dot(self.Z)
        lambda_norm = self.Z.dot(self.Z)

        return diff, lambda_term, lambda_norm

    def calc_grad(self, sig: Sig):
        admm_sig = self.get_Lp()
        return config.beta * (sig - admm_sig) + self.Z
