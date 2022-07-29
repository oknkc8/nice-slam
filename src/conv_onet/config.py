from src.conv_onet import models


def get_model(cfg,  model='nice'):
    """
    Return the network model.

    Args:
        cfg (dict): imported yaml config.
        nice (bool, optional): whether or not use Neural Implicit Scalable Encoding. Defaults to False.

    Returns:
        decoder (nn.module): the network model.
    """

    dim = cfg['data']['dim']
    coarse_grid_len = cfg['grid_len']['coarse']
    middle_grid_len = cfg['grid_len']['middle']
    fine_grid_len = cfg['grid_len']['fine']
    color_grid_len = cfg['grid_len']['color']
    c_dim = cfg['model']['c_dim']  # feature dimensions
    feature_fusion = cfg['model']['feature_fusion']
    pos_embedding_method = cfg['model']['pos_embedding_method']
    if model == 'nice':
        decoder = models.decoder_dict['nice'](
            dim=dim, c_dim=c_dim, coarse=cfg['coarse'], coarse_grid_len=coarse_grid_len,
            middle_grid_len=middle_grid_len, fine_grid_len=fine_grid_len,
            color_grid_len=color_grid_len, pos_embedding_method=pos_embedding_method, feature_fusion=feature_fusion)
    elif model == 'imap':
        decoder = models.decoder_dict['imap'](
            dim=dim, c_dim=0, color=True,
            hidden_size=256, skips=[], n_blocks=4, pos_embedding_method=pos_embedding_method, feature_fusion=feature_fusion
        )
    elif model == 'omni_feat':
        decoder = models.decoder_dict['imap'](
            name='color', dim=dim, c_dim=c_dim, color=True,
            hidden_size=256, skips=[2], n_blocks=5, pos_embedding_method=pos_embedding_method, feature_fusion=feature_fusion
        )
    return decoder
