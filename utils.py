import torch.distributions.multivariate_normal as mn
import numpy as np
import torch

def generate_latent(sample_labels, means, covariances):
    # res = np.array([
    #         np.random.multivariate_normal(means[e], covariances[e])
    #         for e in sample_labels
    #     ])
    # return torch.tensor(res).float()
    # print(len(means[sample_labels[0]]), len(covariances[sample_labels[0]]))
    distribution = mn.MultivariateNormal(means[sample_labels[0]], covariances[sample_labels[0]])
    fake_z = distribution.sample((1,))
    for c in range(1, len(sample_labels[0:])):
        fake_z = torch.cat((fake_z, mn.MultivariateNormal(means[sample_labels[c]], covariances[sample_labels[c]]).sample((1,))), dim=0)
    return fake_z.float()





