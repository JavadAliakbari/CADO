import yaml


class Config:
    def __init__(self, path="config/config.yml"):
        self.config = Config.load_config(path)
        self.mu0 = self.config["mu0"]
        self.mu1 = self.config["mu1"]
        self.mu2 = self.config["mu2"]
        self.iner_iter_GA_R = self.config["iner_iter_GA_R"]
        self.iner_step_GA_R = self.config["iner_step_GA_R"]
        self.iter_or_cycles_to_find_next_atom = self.config[
            "iter_or_cycles_to_find_next_atom"
        ]
        self.rho_min = self.config["rho_min"]
        self.rho_max = self.config["rho_max"]
        self.sig_rho_min = self.config["sig_rho_min"]
        self.sig_rho_max = self.config["sig_rho_max"]
        self.iters = self.config["cogent_iters"]
        self.GP_iters = self.config["GP_iters"]
        self.GP_tol = self.config["GP_tol"]
        self.beta = self.config["beta"]
        self.local_iter = self.config["local_iter"]
        self.Abar = self.config["Abar"]
        self.train_ratio = self.config["train_ratio"]
        self.lr = self.config["lr"]
        self.p = self.config["p"]
        self.q = self.config["q"]
        self.CGD = self.config["CGD"]
        self.warm_start = self.config["warm_start"]
        self.normalize = self.config["normalize"]

        # Load optimizer configurations
        self.optimizer = self._load_optimizer_config()

    def _load_optimizer_config(self):
        """Load optimizer configuration from the config file"""
        optimizer_config = self.config.get("optimizer", {})

        # Default values if not specified in config
        default_optimizer = {
            "type": "adam",
            "learning_rate": {
                "initial": self.lr,  # Use global lr as default
                "min": self.lr * 0.01,
                "decay_rate": 0.95,
                "decay_steps": 100,
            },
            "adam": {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
            "sgd": {"momentum": 0.0},
        }

        # Update defaults with values from config
        for key, value in optimizer_config.items():
            if key in default_optimizer:
                if isinstance(default_optimizer[key], dict) and isinstance(value, dict):
                    default_optimizer[key].update(value)
                else:
                    default_optimizer[key] = value

        return default_optimizer

    @staticmethod
    def load_config(path):
        with open(path) as f:
            config = yaml.load(f, yaml.FullLoader)

        return config
