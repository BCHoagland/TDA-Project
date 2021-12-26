class AE():
    
    def loss(self, batch, model, rec_loss, params):
        enc = model.encode(batch)
        out = model.decode(enc)
        return rec_loss(out, batch)
