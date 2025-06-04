import torch

def MIFGSM_noSPTS(net:torch.nn.Module, adv:torch.Tensor, labels:torch.Tensor, lossfn, step, step_len:torch.Tensor, dev):
    momentum = 0.
    mu = 1.5
    adv.requires_grad_(True)
    alpha = step_len.expand(len(adv), -1)
    
    mask = labels.unsqueeze(1).expand(-1, adv.size(1)) # Dont generate adv flow for benign, using mask to zero pertubation.

    for _ in range(step):
        loss = lossfn(net(adv), labels)
        loss.backward()

        momentum = mu * momentum + adv.grad / torch.norm(adv.grad, p=1)
        adv.data = adv.data + alpha * torch.sign(momentum) * mask

        adv.data = torch.clamp(adv.data, 0., 1.)
        adv.grad.zero_()

    return adv.detach()
