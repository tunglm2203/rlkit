import torch
import torch.nn.functional as F


def vae_loss(vae, batch_data, opt):
    beta = opt.get('beta', 0)
    tgt_latent_distr_param = vae.encode(batch_data)
    latents = vae.reparameterize(tgt_latent_distr_param)
    _, tgt_obs_distr_params = vae.decode(latents)
    log_prob = vae.logprob(batch_data, tgt_obs_distr_params)
    kle = vae.kl_divergence(tgt_latent_distr_param)
    loss = -1 * log_prob + beta * kle
    return loss, log_prob, kle


def vae_loss_rm_rec(vae, batch_data, opt):
    beta = opt.get('beta', 0)
    log_prob = torch.FloatTensor(([0.0]))
    _, tgt_obs_distr_params, tgt_latent_distr_param = vae(batch_data)
    kle = vae.kl_divergence(tgt_latent_distr_param)
    loss = beta * kle
    return loss, log_prob, kle


def vae_loss_stop_grad_dec(vae, batch_data, opt):
    beta = opt.get('beta', 0)
    tgt_latent_distr_param = vae.encode(batch_data)
    latents = vae.reparameterize(tgt_latent_distr_param)
    with torch.no_grad():
        _, tgt_obs_distr_params = vae.decode(latents)

    log_prob = vae.logprob(batch_data, tgt_obs_distr_params)
    kle = vae.kl_divergence(tgt_latent_distr_param)
    loss = -1 * log_prob + beta * kle
    return loss, log_prob, kle


def consistency_loss(pair_sim, pair_real, sim_vae, real_vae, opt):
    use_mu = opt.get('use_mu', False)
    alpha1 = opt.get('alpha1', 0)

    f_distance = opt.get('distance', None)
    assert f_distance is not None, 'Must specify the distance function'

    latent_distribution_params_sim = sim_vae.encode(pair_sim)
    if use_mu:
        _, obs_distribution_params_real = real_vae.decode(latent_distribution_params_sim[0])
    else:
        latents_sim = sim_vae.reparameterize(latent_distribution_params_sim)
        _, obs_distribution_params_real = real_vae.decode(latents_sim)

    latent_distribution_params_real = real_vae.encode(pair_real)
    if use_mu:
        _, obs_distribution_params_sim = sim_vae.decode(latent_distribution_params_real[0])
    else:
        latents_real = real_vae.reparameterize(latent_distribution_params_real)
        _, obs_distribution_params_sim = sim_vae.decode(latents_real)

    ctc_sim2real = f_distance(pair_real, obs_distribution_params_real[0])
    ctc_real2sim = f_distance(pair_sim, obs_distribution_params_sim[0])
    return alpha1 * (ctc_sim2real + ctc_real2sim)


def consistency_loss_w_cycle(pair_sim, pair_real, sim_vae, real_vae, opt):
    ctc_latent_cross = opt.get('ctc_latent_cross', True)
    use_mu = opt.get('use_mu', False)
    alpha1 = opt.get('alpha1', 0)
    alpha2 = opt.get('alpha2', 0)
    alpha3 = opt.get('alpha3', 0)
    f_distance = opt.get('distance', None)
    assert f_distance is not None, 'Must specify the distance function'
    # ============== Consitency loss of image ==============
    latent_params_sim = sim_vae.encode(pair_sim)
    if use_mu:
        _, rec_params_real = real_vae.decode(latent_params_sim[0])
    else:
        latents_sim = sim_vae.reparameterize(latent_params_sim)
        _, rec_params_real = real_vae.decode(latents_sim)

    latent_params_real = real_vae.encode(pair_real)
    if use_mu:
        _, rec_params_sim = sim_vae.decode(latent_params_real[0])
    else:
        latents_real = real_vae.reparameterize(latent_params_real)
        _, rec_params_sim = sim_vae.decode(latents_real)

    ctc_sim2real = f_distance(pair_real, rec_params_real[0], imlength=real_vae.imlength)
    ctc_real2sim = f_distance(pair_sim, rec_params_sim[0], imlength=real_vae.imlength)
    ctc_total = ctc_sim2real + ctc_real2sim

    # ============== Cycle loss of image ==============
    rec_latent_params_real = real_vae.encode(rec_params_real[0].detach())
    _, rec_params_sim_2 = sim_vae.decode(rec_latent_params_real[0])

    rec_latent_params_sim = sim_vae.encode(rec_params_sim[0].detach())
    _, rec_params_real_2 = real_vae.decode(rec_latent_params_sim[0])

    cycle_sim2sim = f_distance(pair_sim, rec_params_sim_2[0])
    cycle_real2real = f_distance(pair_real, rec_params_real_2[0])
    cycle_total = cycle_sim2sim + cycle_real2real

    # ============== Consitency loss of latent ==============
    _, rec_params_real_hat = real_vae.decode(latent_params_sim[0].detach())
    latent_params_real_hat = real_vae.encode(rec_params_real_hat[0])
    if ctc_latent_cross:
        ctc_latent_sim2real = f_distance(latent_params_real[0].detach(), latent_params_real_hat[0])
    else:
        ctc_latent_sim2sim = f_distance(latent_params_sim[0].detach(), latent_params_real_hat[0])

    _, obs_params_sim_hat = sim_vae.decode(latent_params_real[0].detach())
    latent_params_sim_hat = sim_vae.encode(obs_params_sim_hat[0])
    if ctc_latent_cross:
        ctc_latent_real2sim = f_distance(latent_params_sim[0].detach(), latent_params_sim_hat[0])
    else:
        ctc_latent_real2real = f_distance(latent_params_real[0].detach(), latent_params_sim_hat[0])

    if ctc_latent_cross:
        ctc_latent_total = ctc_latent_sim2real + ctc_latent_real2sim
    else:
        ctc_latent_total = ctc_latent_sim2sim + ctc_latent_real2real

    total_loss = alpha1 * ctc_total + alpha2 * cycle_total + alpha3 * ctc_latent_total
    return total_loss


def consistency_loss_rm_dec_path(pair_sim, pair_real, sim_vae, real_vae, opt):
    use_mu = opt.get('use_mu', False)

    latent_distribution_params_real = real_vae.encode(pair_real)
    if use_mu:
        _, obs_distribution_params_sim = sim_vae.decode(latent_distribution_params_real[0])
    else:
        latents_real = real_vae.reparameterize(latent_distribution_params_real)
        _, obs_distribution_params_sim = sim_vae.decode(latents_real)

    ctc_real2sim = sim_vae.logprob(pair_sim, obs_distribution_params_sim)
    return -1 * ctc_real2sim


def mse_pair(pair_sim, pair_real, sim_vae, real_vae):
    src_latent_mean, _ = sim_vae.encode(pair_sim)
    tgt_latent_mean, _ = real_vae.encode(pair_real)
    loss = F.mse_loss(tgt_latent_mean, src_latent_mean)
    return loss

