from grad import Grad


class GraphGrad(Grad):
    def __init__(self, grad_L, grad_Rv, grad_pi) -> None:
        super().__init__()
        self.grad_L = grad_L
        self.grad_Rv = grad_Rv
        self.grad_pi = grad_pi
