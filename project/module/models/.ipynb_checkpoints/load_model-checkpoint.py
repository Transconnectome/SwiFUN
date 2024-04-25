from .swin4d_transformer_ver7 import SwinTransformer4D as SwinTransformer4D_ver7
# from .mae_swin4d_transformer_ver7 import MaskedAutoencoderSwinTransformer4D as MaskedAutoencoderSwinTransformer4D_ver7
# from .mae_swin4d_transformer_ver8 import MaskedAutoencoderSwinTransformer4D as MaskedAutoencoderSwinTransformer4D_ver8
# from .mae_swin4d_transformer_ver9 import MaskedAutoencoderSwinTransformer4D as MaskedAutoencoderSwinTransformer4D_ver9
from monai.networks.nets import SwinUNETR

def load_model(model_name, hparams=None):
    #number of transformer stages
    n_stages = len(hparams.depths)

    if hparams.precision == 16:
        to_float = False
    elif hparams.precision == 32:
        to_float = True

    print(to_float)

    if model_name.lower() == "swinunetr":
        net = SwinUNETR(
            img_size=hparams.img_size,
            in_channels=hparams.sequence_length,
            out_channels=1,
            drop_rate=hparams.attn_drop_rate,
            dropout_path_rate=hparams.attn_drop_rate,
            attn_drop_rate=hparams.attn_drop_rate,
            feature_size=hparams.embed_dim) 
    elif model_name == "emb_mlp":
        from .emb_mlp import mlp
        net = mlp(final_embedding_size=128, num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1)), use_normalization=True)
    elif model_name == "clf_mlp":
        if hparams.clf_head_version == 'v1':
            from .clf_mlp import mlp
            net = mlp(num_classes=2, num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1)))
        elif hparams.clf_head_version == 'v2':
            from .clf_mlp_v2 import mlp
            net = mlp(num_classes=2, num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1)))
        else:
            raise NotImplementedError
        # x -> (b, 96, 4, 4, 4, t)
    elif model_name == "reg_mlp":
        from .clf_mlp import mlp
        net = mlp(num_classes=1, num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1)))
    else:
        raise NameError(f"{model_name} is a wrong model name")

    return net
