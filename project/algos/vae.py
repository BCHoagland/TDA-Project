import torch


class VAE():
    #! TODO: do data_size stuff automatically
    def _KL(self, μ, log_var, batch_size, data_size):
        quit('NEED TO DO BATCH SIZE AND DATA SIZE AUTOMATICALLY')
        kl = -0.5 * torch.sum(1 + log_var - μ.pow(2) - log_var.exp())
        kl /= batch_size * data_size
        return kl

    def loss(self, batch, model, rec_loss, params):
        enc, μ, log_var = model.encode(batch)
        out = model.decode(enc)
        r_loss = rec_loss(out, batch)
        kl_loss = self._KL(μ, log_var)
        return rec_loss + kl_loss
