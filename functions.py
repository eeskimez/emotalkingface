def compute_gradient_penalty(D, real_samples, fake_samples, image_c, z_speech, noise):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1, 1))).to(params['DEVICE'])
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(image_c, interpolates, z_speech, noise)
        fake = Variable(torch.FloatTensor(d_interpolates.shape).fill_(1.0), requires_grad=False).to(params['DEVICE'])
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradient_penalty = 0#gradients[0].view(gradients[0].size(0), -1)
        for grad in gradients:#[1:]:
            gradient_penalty += ((grad.norm(2, dim=1) - 1) ** 2).mean()#torch.cat((grads, grad.view(grad.size(0), -1)), 1)
        # gradient_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

def compute_gradient_penalty_F(D, real_samples, fake_samples, image_c):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1, 1))).to(params['DEVICE'])
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, image_c)
    fake = Variable(torch.FloatTensor(d_interpolates.shape).fill_(1.0), requires_grad=False).to(params['DEVICE'])
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradient_penalty = 0#gradients[0].view(gradients[0].size(0), -1)
    for grad in gradients:#[1:]:
        gradient_penalty += ((grad.norm(2, dim=1) - 1) ** 2).mean()#torch.cat((grads, grad.view(grad.size(0), -1)), 1)
    # gradient_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_grad_P(D, video, speech):
    batch_size = video.size(0)
    d_out = D(video.requires_grad_(True),  speech)
    # print(d_out.shape, torch.ones(d_out.size()).to(params['DEVICE']).shape)
    grad_dout = torch.autograd.grad(
        outputs= d_out, 
        inputs= video,
        grad_outputs= torch.ones(d_out.size()).to(params['DEVICE']),
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True,
    )

    grad_norm = 0
    for grad in grad_dout:
        grad = grad.view(grad.size(0), -1) 
        grad_norm += grad.norm(2, dim=1).mean()
    return grad_norm

    return grad_norm

def compute_grad_F(D, video, image_c):
    batch_size = video.size(0)
    d_out = D(video.requires_grad_(True), image_c)
    # print(d_out.shape, torch.ones(d_out.size()).to(params['DEVICE']).shape)
    grad_dout = torch.autograd.grad(
        outputs= d_out, 
        inputs= video,
        grad_outputs= torch.ones(d_out.size()).to(params['DEVICE']),
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True,
    )
    grad_norm = 0
    for grad in grad_dout:
        grad = grad.view(grad.size(0), -1) 
        grad_norm += grad.norm(2, dim=1).mean()
    return grad_norm