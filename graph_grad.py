from grad import Grad


class GraphGrad(Grad):
    def __init__(self, grad_L, grad_Rv, grad_pi) -> None:
        super().__init__()
        self.grad_L = grad_L
        self.grad_Rv = grad_Rv
        self.grad_pi = grad_pi
        self.n = grad_L.shape[0]
        self.d = grad_Rv.shape[-1]
        self.c = self.grad_pi.shape[-1]

    def get_params(self):
        return self.n, self.d, self.c
