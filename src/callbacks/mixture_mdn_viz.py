import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

def _chol_log_prob_full(y, mu, L):
    """
    Log N(y | mu, LL^T) for batched arrays.
    Shapes: y(B,d), mu(B,K,d), L(B,K,d,d) lower-tri with positive diag.
    Returns: log_prob (B,K)
    """
    B, K, d = mu.shape
    y_exp = y.unsqueeze(1)             # (B,1,d)
    diff  = (y_exp - mu).unsqueeze(-1) # (B,K,d,1)

    # enforce lower tri + positive diag before solve (should already be true)
    L = torch.tril(L)
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    diag_pos = torch.nn.functional.softplus(diag) + 1e-6
    L = L - torch.diag_embed(diag) + torch.diag_embed(diag_pos)

    z = torch.linalg.solve_triangular(L.float(), diff.float(), upper=False)   # (B,K,d,1)
    maha = (z.square().sum(dim=(-2, -1))).to(mu.dtype)                        # (B,K)
    logdet = (2.0 * diag_pos.float().log().sum(dim=-1)).to(mu.dtype)          # (B,K)
    const = d * math.log(2.0 * math.pi)
    return -0.5 * (const + logdet + maha)                                     # (B,K)

def _diag_log_prob(y, mu, sigma):
    """
    Log N(y|mu, diag(sigma^2)).
    Shapes: y(B,d), mu(B,K,d), sigma(B,K,d) positive
    Returns: (B,K)
    """
    var = (sigma * sigma).clamp_min(1e-12)
    d = mu.shape[-1]
    const = d * math.log(2.0 * math.pi)
    log_det = var.log().sum(dim=-1)                 # (B,K)
    maha = ((y.unsqueeze(1) - mu).square() / var).sum(dim=-1)  # (B,K)
    return -0.5 * (const + log_det + maha)

def _mixture_marginals_diag(pi_logits, mu, sigma):
    pi = torch.softmax(pi_logits, dim=-1)           # (B,K)
    mu_mix = (pi.unsqueeze(-1) * mu).sum(dim=1)     # (B,d)
    second = (pi.unsqueeze(-1) * (sigma*sigma + mu*mu)).sum(dim=1)
    var_mix = (second - mu_mix*mu_mix).clamp_min(1e-12)
    return mu_mix, var_mix.sqrt()

def _mixture_marginals_full(pi_logits, mu, L):
    pi = torch.softmax(pi_logits, dim=-1)           # (B,K)
    mu_mix = (pi.unsqueeze(-1) * mu).sum(dim=1)     # (B,d)
    diagSigma_k = (L * L).sum(dim=-1)               # (B,K,d)
    second = (pi.unsqueeze(-1) * (diagSigma_k + mu*mu)).sum(dim=1)
    var_mix = (second - mu_mix*mu_mix).clamp_min(1e-12)
    return mu_mix, var_mix.sqrt()

def _ellipse_from_cov_2d(mu2, Sigma2, n_std=1.0, num=200):
    # returns (xs, ys) for plotting
    w, v = np.linalg.eigh(Sigma2)
    w = np.clip(w, 1e-12, None)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    ang = np.arctan2(v[1,0], v[0,0])
    t = np.linspace(0, 2*np.pi, num)
    ell = np.stack([np.cos(t), np.sin(t)], axis=0)
    scale = n_std * np.sqrt(w)
    R = np.array([[np.cos(ang), -np.sin(ang)],
                  [np.sin(ang),  np.cos(ang)]])
    pts = (R @ (ell * scale[:,None])).T + mu2[None,:]
    return pts[:,0], pts[:,1]

class MixtureMDNVizCallback(Callback):
    def __init__(self, param_pairs=None, max_ellipse_samples=50, output_dir="./pred_plots", job_id="default"):
        super().__init__()
        self.param_pairs = param_pairs  # list of tuples of param names, e.g. [("D","L"), ("rr","ro")]
        self.max_ellipse_samples = max_ellipse_samples
        self.output_dir = output_dir
        self.job_id = job_id

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        cache = getattr(pl_module, "_val_cache", {})
        y = cache.get("targets", None)              # (N,d) standardized
        pi_logits = cache.get("pi_logits_all", None)# (N,K)
        mu_all = cache.get("mu_all", None)          # (N,K,d)
        L_all  = cache.get("L_all", None)           # (N,K,d,d) or None
        sigma_all = cache.get("sigma_all", None)    # (N,K,d) or None

        print(f"[DEBUG] MixtureMDNVizCallback: cached y: {y.shape if y is not None else None}, pi_logits: {pi_logits if pi_logits is not None else None}, mu_all: {mu_all.shape if mu_all is not None else None}, L_all: {L_all.shape if L_all is not None else None}, sigma_all: {sigma_all.shape if sigma_all is not None else None}", flush=True)
        if y is None or pi_logits is None or mu_all is None:
            print("MixtureMDNVizCallback: No cached data found, skipping viz.", flush=True)
            return  # Not a mixture run or nothing cached yet

        y = y.float()
        pi_logits = pi_logits.float()
        mu_all = mu_all.float()
        K = pi_logits.shape[1]
        d = mu_all.shape[-1]

        # choose full or diag log-prob
        if L_all is not None:
            L_all = L_all.float()
            log_prob = _chol_log_prob_full(y, mu_all, L_all)   # (N,K)
        else:
            sigma_all = sigma_all.float().clamp_min(1e-6)
            log_prob = _diag_log_prob(y, mu_all, sigma_all)    # (N,K)

        log_pi = torch.log_softmax(pi_logits, dim=-1)
        log_post = log_pi + log_prob
        r = torch.softmax(log_post, dim=-1)   # responsibilities (N,K)
        hard = torch.argmax(r, dim=-1).cpu().numpy()

        os.makedirs(f"{self.output_dir}/Mixture/", exist_ok=True)
        base = f"{self.output_dir}/Mixture/"

        # 1) mixture prior histogram
        plt.figure(figsize=(6,4))
        pi = torch.softmax(pi_logits, dim=-1).detach().cpu().numpy()
        for k in range(K):
            plt.hist(pi[:,k], bins=30, alpha=0.5, label=f"π{k}")
        plt.xlabel("π value"); plt.ylabel("count"); plt.legend(); plt.tight_layout()
        plt.savefig(f"{base}/pi_hist.png", dpi=140); plt.close()

        # 2) hard assignment histogram
        plt.figure(figsize=(5,4))
        bins = np.arange(K+1)-0.5
        plt.hist(hard, bins=bins, rwidth=0.9)
        plt.xticks(range(K))
        plt.xlabel("argmax responsibility (component)"); plt.ylabel("count"); plt.tight_layout()
        plt.savefig(f"{base}/hard_assign_hist.png", dpi=140); plt.close()

        # 3) Mahalanobis chi^2 Q–Q (using responsible component)
        idx = torch.arange(y.shape[0])
        if L_all is not None:
            L_resp = L_all[idx, hard]     # (N,d,d)
            mu_resp = mu_all[idx, hard]   # (N,d)
            # enforce positivity
            Lr = torch.tril(L_resp)
            diag = torch.diagonal(Lr, dim1=-2, dim2=-1)
            Lr = Lr - torch.diag_embed(diag) + torch.diag_embed(torch.nn.functional.softplus(diag)+1e-6)
            diff = (y - mu_resp).unsqueeze(-1)
            z = torch.linalg.solve_triangular(Lr.float(), diff.float(), upper=False).squeeze(-1)
            d2 = (z*z).sum(dim=-1).cpu().numpy()
        else:
            sigma_resp = sigma_all[idx, hard]  # (N,d)
            var = (sigma_resp*sigma_resp).clamp_min(1e-12)
            d2 = (((y - mu_all[idx, hard])**2 / var).sum(dim=-1)).cpu().numpy()

        # Q–Q vs chi2_d
        from scipy.stats import chi2
        q_emp = np.quantile(d2, np.linspace(0.01, 0.99, 99))
        q_the = chi2.ppf(np.linspace(0.01, 0.99, 99), df=d)
        plt.figure(figsize=(5,5))
        plt.plot(q_the, q_emp, 'o', ms=3)
        lo, hi = min(q_the.min(), q_emp.min()), max(q_the.max(), q_emp.max())
        plt.plot([lo,hi],[lo,hi],'r--')
        plt.xlabel("Theoretical χ² quantiles"); plt.ylabel("Empirical quantiles")
        plt.title(f"Q–Q of Mahalanobis (d={d})")
        plt.tight_layout(); plt.savefig(f"{base}/qq_mahalanobis.png", dpi=150); plt.close()

        # 4) Mixture mean ± mixture σ per-parameter (pred vs true, original scale)
        ds = getattr(trainer.datamodule, "dataset_ref", None)
        param_names = list(getattr(ds, "model_params", [f"p{i}" for i in range(d)]))
        means = getattr(ds, "scaler_means", None)
        stds  = getattr(ds, "scaler_stds", None)
        if means is not None and stds is not None:
            means = means.detach().cpu().numpy()
            stds  = stds.detach().cpu().numpy()

            if L_all is not None:
                mu_mix, sig_mix = _mixture_marginals_full(pi_logits, mu_all, L_all)
            else:
                mu_mix, sig_mix = _mixture_marginals_diag(pi_logits, mu_all, sigma_all)
            mu_mix = mu_mix.detach().cpu().numpy()
            sig_mix = sig_mix.detach().cpu().numpy()

            y_true = y.detach().cpu().numpy()

            # inverse scale
            y_true = y_true * stds + means
            mu_mix = mu_mix * stds + means
            sig_mix = sig_mix * stds  # covariance scales with std; diag uses *std

            os.makedirs(f"{base}/per_param", exist_ok=True)
            for j, name in enumerate(param_names):
                plt.figure(figsize=(6,6))
                plt.errorbar(y_true[:,j], mu_mix[:,j], yerr=sig_mix[:,j], fmt='o', alpha=0.25, markersize=3, capsize=2)
                mn = min(y_true[:,j].min(), mu_mix[:,j].min())
                mx = max(y_true[:,j].max(), mu_mix[:,j].max())
                plt.plot([mn,mx],[mn,mx],'r--')
                plt.xlabel(f"True {name}")
                plt.ylabel(f"Pred (mixture mean) {name}")
                plt.title(f"{name}: mixture mean ± σ_mix")
                plt.tight_layout()
                plt.savefig(f"{base}/per_param/{name}_mixmean.png", dpi=140)
                plt.close()

        # 5) 2D ellipses for a param pair (original scale)
        if self.param_pairs and (L_all is not None or sigma_all is not None):
            means = getattr(ds, "scaler_means", None)
            stds  = getattr(ds, "scaler_stds", None)
            if means is None or stds is None:
                return
            means = means.detach().cpu().numpy()
            stds  = stds.detach().cpu().numpy()
            name_to_idx = {n:i for i,n in enumerate(param_names)}

            # choose a subset
            N = y.shape[0]
            sel = np.random.choice(N, size=min(self.max_ellipse_samples, N), replace=False)

            for (a,b) in self.param_pairs:
                ia, ib = name_to_idx[a], name_to_idx[b]
                plt.figure(figsize=(7,7))
                # plot truths
                y_np = y.detach().cpu().numpy()
                y_ab = y_np[:, [ia, ib]] * stds[[ia,ib]] + means[[ia,ib]]
                plt.scatter(y_ab[sel,0], y_ab[sel,1], s=8, alpha=0.35, label="truths")

                for n in sel:
                    pis = torch.softmax(pi_logits[n], dim=-1).cpu().numpy()
                    # use responsibilities to modulate alpha
                    r_n = r[n].cpu().numpy()

                    for k in range(K):
                        if L_all is not None:
                            # build 2x2 covariance in original scale:
                            # L_real = S * L_norm  (S = diag(stds))
                            Lnorm = L_all[n,k].detach().cpu().numpy()
                            S = np.diag(stds)
                            Lreal = S @ Lnorm
                            Sigma = Lreal @ Lreal.T
                            mu = (mu_all[n,k].detach().cpu().numpy() * stds + means)
                            mu2 = mu[[ia,ib]]
                            Sigma2 = Sigma[[ia,ib]][:,[ia,ib]]
                        else:
                            # diag case: axis-aligned ellipse from sigma
                            s = (sigma_all[n,k].detach().cpu().numpy() * stds)
                            mu = (mu_all[n,k].detach().cpu().numpy() * stds + means)
                            mu2 = mu[[ia,ib]]
                            Sigma2 = np.diag(s[[ia,ib]]**2)

                        # Draw 1σ and 2σ
                        for ns in (1.0, 2.0):
                            xs, ys = _ellipse_from_cov_2d(mu2, Sigma2, n_std=ns)
                            plt.plot(xs, ys, alpha=0.15 + 0.55*r_n[k], lw=1)

                plt.xlabel(a); plt.ylabel(b)
                plt.title(f"Per-sample component ellipses (1σ/2σ): {a} vs {b}")
                plt.legend(loc="best")
                plt.tight_layout()
                plt.savefig(f"{base}/ellipses_{a}_vs_{b}.png", dpi=150)
                plt.close()
