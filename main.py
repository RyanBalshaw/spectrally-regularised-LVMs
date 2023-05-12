"""
Docstring
"""

import os

import numpy as np
from matplotlib import pyplot as plt

from spectrally_constrained_LVMs import (
    Hankel_matrix,
    linear_model,
    negentropy_cost,
    sympy_cost,
    user_cost,
)

if __name__ == "__main__":
    Lw = 256
    Lsft = 1
    source_name = "exp"

    tmp_dir_path = os.path.join(os.environ["TMP_DIR"], ("sica_") + source_name)

    # # Phenomenological model signal
    data_dir = os.path.join(
        os.environ["DATA_DIR"],
        "singleFiles",
        # "cs2_imp_files",
        "ica_pheno_const_bd.npy",  # "ica_pheno_const_A3_1.npy",  # "ica_pheno_const_analysis.npy"
    )
    Fs = 25e3
    data_dict = np.load(data_dir, allow_pickle=True).item()
    x_signal = data_dict["x_total"]

    # plt.figure()
    # n = len(x_signal)
    # freq = np.fft.fftfreq(n, 1/Fs)[:n//2]
    # val = 2/n * np.abs(np.fft.fft(x_signal))[:n // 2]
    #
    # plt.plot(freq, val)
    # plt.show(block = True)

    # IMS dataset
    # data_dir = os.path.join(
    #     os.environ["DATA_DIR"], "Datasets", "IMS", "IMS_Dataset2")#, "2004.02.17.16.02.39" # "2004.02.16.05.12.39" # "2004.02.16.03.32.39" #
    #     #"2004.02.12.10.32.39" # "2004.02.17.17.52.39" #
    # #)
    # Fs = 20480
    # files = sorted(os.listdir(data_dir))
    # data_dict = np.loadtxt(os.path.join(data_dir, files[540]))
    # x_signal = data_dict[:, 0]

    #######################################
    X = Hankel_matrix(x_signal, Lw, Lsft)

    # cost_inst = negentropy_cost(source_name = source_name,
    #                             source_params = {"alpha": 1})

    cost_inst = user_cost(use_hessian=True)

    # linear_model = lambda X, w: X @ w
    loss = (
        lambda X, w, y: -1 * np.mean(y**2, axis=0) / (w.T @ w)[0, 0]
    )  # -1 * np.mean(y ** 2, axis=0)
    grad = lambda X, w, y: -2 * (
        (np.cov(X, rowvar=False) @ w) / (w.T @ w)
        - np.mean(y**2, axis=0) / ((w.T @ w) ** 2) * w
    )  # -2 * np.mean(y * X, axis=0, keepdims=True).T
    hess = lambda X, w, y: -2 * (
        (np.cov(X, rowvar=False) / (w.T @ w))
        + (-2 / ((w.T @ w) ** 2) * (np.cov(X, rowvar=False) @ w) @ w.T)
        - 1
        * (
            (np.mean(y**2, axis=0) / (w.T @ w) * np.eye(w.shape[0]))
            + (
                w
                @ np.transpose(
                    2 * (np.cov(X, rowvar=False) @ w) / ((w.T @ w) ** 2)
                    - 4 * np.mean(y**2, axis=0) / ((w.T @ w) ** 3) * w
                )
            )
        )
    )
    # -2 * np.cov(X, rowvar=False) # (X.T @ X) / X.shape[0]

    cost_inst.set_cost(loss)
    cost_inst.set_gradient(grad)
    cost_inst.set_hessian(hess)

    sICA_inst = linear_model(
        n_sources=256,
        cost_instance=cost_inst,
        whiten=False,
        init_type="broadband",
        organise_by_kurtosis=False,
        perform_gso=True,
        batch_size=None,
        var_PCA=None,
        alpha_reg=0,  # 0.0005,
        sumt_flag=False,
        sumt_parameters={
            "eps_1": 1e-4,
            "alpha_init": 1,
            "alpha_end": 100,
            "alpha_multiplier": 10,
        },
        jacobian_update_type="full",
        use_ls=True,
        use_hessian=True,
        save_dir=tmp_dir_path,
        verbose=True,
    )

    print("Fitting the model...")

    sICA_inst.fit(X, n_iters=500, learning_rate=1, tol=1e-4, approx_flag=True, Fs=Fs)
