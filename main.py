"""
Docstring
"""

import os

import numpy as np
import sympy as sp

from spectrally_constrained_LVMs import (
    Hankel_matrix,
    linear_model,
    negentropy_cost,
    sympy_cost,
    user_cost,
    variance_cost,
)

# from matplotlib import pyplot as plt


if __name__ == "__main__":
    Lw = 512
    Lsft = 1
    source_name = "exp"

    tmp_dir_path = os.path.join(os.environ["TMP_DIR"], ("sica_") + source_name)

    # # Phenomenological model signal
    data_dir = os.path.join(
        os.environ["DATA_DIR"],
        "singleFiles",
        # "cs2_imp_files",
        "ica_pheno_const_bd.npy",
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
    #     os.environ["DATA_DIR"], "Datasets", "IMS", "IMS_Dataset2")
    # #)
    # Fs = 20480
    # files = sorted(os.listdir(data_dir))
    # data_dict = np.loadtxt(os.path.join(data_dir, files[540]))
    # x_signal = data_dict[:, 0]

    #######################################
    X = Hankel_matrix(x_signal, Lw, Lsft)
    # X = X[: X.shape[0] // 12, :]
    # print(X.shape)

    # Method 1
    # n_samples = X.shape[0]
    # n_features = Lw
    #
    # test_inst = sympy_cost(n_samples, n_features, use_hessian=False)
    #
    # X_sp, w_sp, iter_params = test_inst.get_model_parameters()
    # i, j = iter_params
    #
    # loss_i = sp.Sum(w_sp[j, 0] * X_sp[i, j], (j, 0, n_features - 1))
    # loss = -1 / n_samples * sp.Sum((loss_i) ** 2, (i, 0, n_samples - 1))
    #
    # test_inst.set_cost(loss)
    # test_inst.implement_methods()

    # Method 2
    # test_inst = user_cost(use_hessian=False)
    #
    # def loss(X, w, y):
    #     return -1 * np.mean((X @ w) ** 2, axis=0)
    #
    # def grad(X, w, y):
    #     return -2 * np.mean(y * X, axis=0, keepdims=True).T
    #
    # def hess(X, w, y):
    #     return -2 * np.cov(X, rowvar=False)
    #
    # test_inst.set_cost(loss)
    # test_inst.set_gradient(grad)
    # test_inst.set_hessian(hess)

    # Method 3
    test_inst = negentropy_cost(source_name="exp", source_params={"alpha": 1})

    # Method 4
    # test_inst = variance_cost(use_hessian=False, verbose=True)

    # Alternative PCA cost function.
    # linear_model = lambda X, w: X @ w
    # loss = (
    #     lambda X, w, y: -1 * np.mean(y**2, axis=0) / (w.T @ w)[0, 0]
    # )  #
    #
    # grad = lambda X, w, y: -2 * (
    #     (np.cov(X, rowvar=False) @ w) / (w.T @ w)
    #     - np.mean(y**2, axis=0) / ((w.T @ w) ** 2) * w
    # )
    # hess = lambda X, w, y: -2 * (
    #     (np.cov(X, rowvar=False) / (w.T @ w))
    #     + (-2 / ((w.T @ w) ** 2) * (np.cov(X, rowvar=False) @ w) @ w.T)
    #     - 1
    #     * (
    #         (np.mean(y**2, axis=0) / (w.T @ w) * np.eye(w.shape[0]))
    #         + (
    #             w
    #             @ np.transpose(
    #                 2 * (np.cov(X, rowvar=False) @ w) / ((w.T @ w) ** 2)
    #                 - 4 * np.mean(y**2, axis=0) / ((w.T @ w) ** 3) * w
    #             )
    #         )
    #     )
    # )
    #

    sICA_inst = linear_model(
        n_sources=5,
        cost_instance=test_inst,
        whiten=True,
        init_type="broadband",
        organise_by_kurt=True,
        perform_gso=True,
        batch_size=None,
        var_PCA=None,
        alpha_reg=10,  # 0.0005,
        sumt_flag=True,
        sumt_parameters={
            "eps_1": 1e-4,
            "alpha_init": 1,
            "alpha_end": 100,
            "alpha_multiplier": 10,
        },
        hessian_update_type="full",
        use_ls=True,
        use_hessian=True,
        save_dir=tmp_dir_path,
        verbose=True,
    )

    print("Fitting the model...")

    sICA_inst.fit(X, n_iters=500, learning_rate=1, tol=1e-4, Fs=Fs)
