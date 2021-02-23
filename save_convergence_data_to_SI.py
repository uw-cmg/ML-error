import numpy as np

# specify datasets to run -- choices = ["Diffusion", "Friedman_500", "Perovskite"]
datasets = ["Perovskite"]
# specify models to run -- choices = ["RF", "LR", "GPR"]
models = ["RF", "LR"]

for dataset in datasets:
    for model in models:

        a_nll = np.load('data_for_paper_plots/{}/{}/Convergence/a_nll.npy'.format(dataset, model))
        b_nll = np.load('data_for_paper_plots/{}/{}/Convergence/b_nll.npy'.format(dataset, model))

        print(a_nll)
        print(b_nll)

        np.savetxt("SI/{}/{}/Convergence/a_convergence.csv".format(dataset, model), a_nll, header="number_of_models, mu, sigma", delimiter=",")
        np.savetxt("SI/{}/{}/Convergence/b_convergence.csv".format(dataset, model), b_nll, header="number_of_models, mu, sigma",
                   delimiter=",")