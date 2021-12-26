import torch


class VAE():
    #! TODO: do data_size stuff automatically
    def _KL(self, μ, log_var, batch_size, data_size):
        kl = -0.5 * torch.sum(1 + log_var - μ.pow(2) - log_var.exp())
        kl /= batch_size * data_size
        return kl

    
    def loss(self, batch, model, config):
        enc, μ, log_var = model.encode(batch)
        out = model.decode(enc)
        rec_loss = config.rec_loss(out, batch)
        kl_loss = self._KL(μ, log_var, config.batch_size, config.data_size)
        return rec_loss + kl_loss
