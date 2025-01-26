from utils.utils import *


class ADMM_cls:
    def __init__(self, L, constrint_func=project_to_bound):
        self.projection_func = constrint_func
        self.Lambda = np.zeros(L.shape)
        self.Lp = L
        self.Lp = constrint_func(self.Lp)
        self.Lp_prev = None

    def adaptive_admm_update(primal_residual, dual_residual, rho, alpha=1.5):
        """
        Update the penalty parameter rho adaptively based on the primal and dual residuals.

        Args:
            primal_residual (float): The primal residual.
            dual_residual (float): The dual residual.
            rho (float): The current penalty parameter.
            alpha (float): The scaling factor for adaptive update.

        Returns:
            float: The updated penalty parameter rho.
        """
        if primal_residual > 10 * dual_residual:
            rho *= alpha
        elif dual_residual > 10 * primal_residual:
            rho /= alpha
        return rho

    def admm_value(self, L):
        n = L.shape[1]
        d = L - self.Lp
        primal_term = matrix_dot_product(d, d) / n
        lambda_term = matrix_dot_product(self.Lambda, d) / n
        return lambda_term + config.beta / 2 * primal_term

    def ADMM_update(self, L):
        # min Lp
        self.Lp_prev = self.Lp
        self.Lp = self.Lambda / config.beta + L
        self.Lp = self.projection_func(self.Lp)

        # max Lambda
        d = L - self.Lp
        self.Lambda += config.beta * d
        diff = matrix_dot_product(d, d)

        lambda_term = matrix_dot_product(self.Lambda, d)

        # # Adaptive ADMM update
        # primal_residual = np.linalg.norm(d)
        # dual_residual = np.linalg.norm(config.beta * (Lp - Lp_prev))
        # config.beta = adaptive_admm_update(
        #     primal_residual, dual_residual, config.beta
        # )

        return diff, lambda_term

    def calc_grad(self, L):
        return L - self.Lp + self.Lambda / config.beta
