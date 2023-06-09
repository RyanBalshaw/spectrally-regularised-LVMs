"""
Docstring
"""

import os

import numpy as np

from spectrally_regularised_LVMs import LinearModel, NegentropyCost, hankel_matrix

# from matplotlib import pyplot as plt


if __name__ == "__main__":
    Lw = 256
    Lsft = 1
    source_name = "exp"

    tmp_dir_path = os.path.join(os.environ["TMP_DIR"], ("sica_") + source_name)

    # Phenomenological model signal
    # data_dir = os.path.join(
    #     os.environ["DATA_DIR"],
    #     "singleFiles",
    #     # "cs2_imp_files",
    #     "ica_pheno_const_bd.npy",
    # )
    # Fs = 25e3
    # data_dict = np.load(data_dir, allow_pickle=True).item()
    # x_signal = data_dict["x_total"]

    # plt.figure()
    # n = len(x_signal)
    # freq = np.fft.fftfreq(n, 1/Fs)[:n//2]
    # val = 2/n * np.abs(np.fft.fft(x_signal))[:n // 2]
    #
    # plt.plot(freq, val)
    # plt.show(block = True)

    # IMS dataset
    data_dir = os.path.join(os.environ["DATA_DIR"], "Datasets", "IMS", "IMS_Dataset2")
    # )
    Fs = 20480
    files = sorted(os.listdir(data_dir))

    X_ = []
    for i in range(1):
        data_dict = np.loadtxt(os.path.join(data_dir, files[i]))
        x_signal = data_dict[:, 0]

        X_.append(hankel_matrix(x_signal, Lw, Lsft))

    X = np.vstack(X_)

    # Gearbox signal
    # data_dir = os.path.join(
    #     os.environ["DATA_DIR"],
    #     "Datasets",
    #     "Gearbox",
    #     "Exp1Ds",
    #     "Exp1D_R1336.mat"
    # )  # r"D:\PhD_Files\Datasets\Gearbox\Exp1Ds\Exp1D_R1400.mat"
    # Fs = 25.6e3
    # data_mat = io.loadmat(data_dir)
    # x_signal = data_mat["Track1"][0, :]
    # x_signal = x_signal[:len(x_signal) // 2]

    #######################################
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
    test_inst = NegentropyCost(source_name="exp", source_params={"alpha": 1})

    # Method 4
    # test_inst = VarianceCost(use_hessian=True, verbose=True)

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

    # def l1_l2_norm(X, w, y):
    #
    #     N = X.shape[0]
    #
    #     E1 = np.mean(np.abs(y), axis = 0)
    #     E2 = np.mean(y**2, axis = 0)
    #
    #     return np.sqrt(N) * E1 / np.sqrt(E2)
    #
    # def l1_l2_grad(X, w, y):
    #
    #     N = X.shape[0]
    #
    #     E1 = np.mean(np.abs(y), axis=0)
    #     E2 = np.mean(y ** 2, axis=0)
    #
    #     p1 = np.mean(np.sign(y) * X, axis = 0, keepdims=True).T / np.sqrt(E2)
    #     p2 = -1 * E1 / ((E2)**(3/2)) * np.mean(y * X, axis = 0, keepdims=True).T
    #
    #     return np.sqrt(N)  * (p1 + p2)
    #
    #
    # def l1_l2_hessian(X, w, y):
    #     pass

    # test_inst = user_cost(use_hessian=False)
    #
    # test_inst.set_cost(l1_l2_norm)
    # test_inst.set_gradient(l1_l2_grad)

    sICA_inst = LinearModel(
        n_sources=20,
        cost_instance=test_inst,
        whiten=True,
        init_type="broadband",
        perform_gso=True,
        batch_size=None,
        var_PCA=None,
        alpha_reg=1,  # 0.0005,
        sumt_flag=False,
        sumt_parameters={
            "eps_1": 1e-4,
            "alpha_init": 0.1,
            "alpha_end": 10,
            "alpha_multiplier": 10,
        },
        organise_by_kurt=True,
        hessian_update_type="actual",
        use_ls=True,
        use_hessian=True,
        save_dir=None,
        verbose=True,
    )

    print("Fitting the model...")

    sICA_inst.fit(X, n_iters=500, learning_rate=1, tol=1e-4, Fs=Fs)


# def compute_LHIs(data_dir, files):
#     LHIs = []
#     RMSs = []
#
#     for i in range(0, len(files), 5):  # len(files):
#         data_dict = np.loadtxt(os.path.join(data_dir, files[i]))
#         x_signal = data_dict[:, 0]
#
#         X_eval = hankel_matrix(x_signal, Lw, Lsft)
#
#         S = sICA_inst.transform(X_eval)
#
#         LHI_i = np.sqrt(np.sum(S**2, axis=1))
#
#         RMS_i = np.sqrt(1 / len(LHI_i) * np.sum(LHI_i**2))
#
#         LHIs.append(LHI_i)
#         RMSs.append(RMS_i)
#
#     return LHIs, RMSs
#
#
# def fft_vis(signal, Fs, Nfft=None):
#     if Nfft is None:
#         Nfft = len(signal)
#
#     n = len(signal)
#     freq = np.fft.fftfreq(n, 1 / Fs)[: n // 2]
#     val = 2 / n * np.abs(np.fft.fft(signal))[: n // 2]
#
#     return freq, val
