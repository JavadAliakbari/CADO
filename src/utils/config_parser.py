import yaml


class Config:
    def __init__(self, path="config/config.yml"):
        self.config = Config.load_config(path)
        # Load data configurations
        self.data = DataConfig(self.config["data"])
        # Load algorithm configurations
        self.algorithm = AlgorithmConfig(self.config["algorithm"])
        # Load optimizer configurations
        self.optimizer = OptimizerConfig(self.config["optimizer"])

    @staticmethod
    def load_config(path):
        with open(path) as f:
            config = yaml.load(f, yaml.FullLoader)
        return config


class OptimizerConfig:
    def __init__(self, config):
        self.type = config["type"]
        self.lr = config["learning_rate"]["initial"]
        self.min = config["learning_rate"]["min"]
        self.decay_rate = config["learning_rate"]["decay_rate"]
        self.decay_steps = config["learning_rate"]["decay_steps"]
        self.beta1 = config["adam"]["beta1"]
        self.beta2 = config["adam"]["beta2"]
        self.epsilon = config["adam"]["epsilon"]
        self.momentum = config["sgd"]["momentum"]
        self.method = config["learning_rate"]["method"]


class DataConfig:
    def __init__(self, config):
        self.samples_per_cluster = config["samples_per_cluster"]
        self.num_features = config["num_features"]
        self.num_signal_features = config["num_signal_features"]
        self.train_ratio = config["train_ratio"]
        self.p = config["p"]
        self.q = config["q"]
        self.mu0 = config["mu0"]
        self.mu1 = config["mu1"]
        self.mu2 = config["mu2"]
        self.rho_min = config["rho_min"]
        self.rho_max = config["rho_max"]
        self.sig_rho_min = config["sig_rho_min"]
        self.sig_rho_max = config["sig_rho_max"]


class AlgorithmConfig:
    def __init__(self, config):
        self.warmup_steps = config["warmup_steps"]
        self.momentum_beta = config["momentum_beta"]
        self.mu_lr_max = config["mu_lr_max"]
        self.mu_lr_min = config["mu_lr_min"]
        self.min_mu = config["min_mu"]
        self.soft_assign_s = config["soft_assign_s"]
        self.iters = config["iters"]
        self.Abar = config["Abar"]
        self.warm_start = config["warm_start"]
        self.normalize = config["normalize"]
