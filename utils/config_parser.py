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

        self.tau = self.config["tau"]
        self.eta = self.config["eta"]
        self.cogent_iters = self.config["cogent_iters"]
        self.noise_level_Ri = self.config["noise_level_Ri"]
        self.include_rhos = self.config["include_rhos"]
        self.GP_iters = self.config["GP_iters"]
        self.GP_tol = self.config["GP_tol"]
        self.alpha = self.config["alpha"]
        self.beta = self.config["beta"]
        self.line_search = self.config["line_search"]
        self.local_iter = self.config["local_iter"]
        self.eps = self.config["eps"]
        self.enhance = self.config["Enhance"]
        self.Abar = self.config["Abar"]
        self.ADMM = self.config["ADMM"]
        self.train_ratio = self.config["train_ratio"]
        self.lr = self.config["lr"]
        self.p = self.config["p"]
        self.q = self.config["q"]
        self.signal_subspace_dim = self.config["signal_subspace_dim"]
        self.num_nodes = self.config["num_nodes"]
        self.dimension = self.config["dimension"]
        self.num_clusters = self.config["num_clusters"]
        self.samples_per_cluster = self.config["samples_per_cluster"]
        self.total_iterations = self.config["total_iterations"]
        self.local_iter_ALP = self.config["local_iter_ALP"]
        self.local_iter_ALM = self.config["local_iter_ALM"]
        self.initialize_atoms = self.config["initialize_atoms"]

    def load_config(path):
        with open(path) as f:
            config = yaml.load(f, yaml.FullLoader)

        return config
