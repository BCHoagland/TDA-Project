class AE():
    
    def loss(self, batch, model, config):
        enc = model.encode(batch)
        out = model.decode(enc)
        return config.rec_loss(out, batch)
