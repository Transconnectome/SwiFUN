from .swin4d_transformer_ver7 import SwinTransformer4D as SwinTransformer4D_ver7
from .swinunetr_og import SwinUNETR, SwinUNETR_encoder
from .brain_surf_cnn import BrainSurfCNN3D

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
            out_channels=hparams.out_chans,
            patch_size=hparams.patch_size,
            depths=hparams.depths,
            window_size=hparams.window_size,
            drop_rate=hparams.attn_drop_rate,
            dropout_path_rate=hparams.attn_drop_rate,
            attn_drop_rate=hparams.attn_drop_rate,
            feature_size=hparams.embed_dim,
            use_v2=hparams.use_v2) 
    elif model_name.lower() == "swinunetr_encoder":
        net = SwinUNETR_encoder(
            img_size=hparams.img_size,
            in_channels=hparams.sequence_length,
            out_channels=hparams.out_chans,
            patch_size=hparams.patch_size,
            depths=hparams.depths,
            window_size=hparams.window_size,
            drop_rate=hparams.attn_drop_rate,
            dropout_path_rate=hparams.attn_drop_rate,
            attn_drop_rate=hparams.attn_drop_rate,
            feature_size=hparams.embed_dim,
            use_v2=hparams.use_v2)
    elif model_name.lower() == "swin4d_ver7":
        net = SwinTransformer4D_ver7(
            img_size=hparams.img_size,
            in_chans=hparams.in_chans,
            embed_dim=hparams.embed_dim,
            window_size=hparams.swift_window_size,
            first_window_size=hparams.swift_window_size,
            patch_size=hparams.swift_patch_size,
            depths=hparams.depths,
            num_heads=hparams.num_heads,
            c_multiplier=hparams.c_multiplier,
            last_layer_full_MSA=hparams.last_layer_full_MSA,
            to_float = to_float,
            drop_rate=hparams.attn_drop_rate,
            drop_path_rate=hparams.attn_drop_rate,
            attn_drop_rate=hparams.attn_drop_rate
        )
    elif model_name.lower() == "brainsurfcnn3d":
        net = BrainSurfCNN3D(in_ch=hparams.sequence_length, out_ch=1)
    elif model_name == "emb_mlp":
        from .emb_mlp import mlp
        net = mlp(final_embedding_size=128, num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1)), use_normalization=True)
    elif model_name == "emb_mlp_swin4d":
        from .emb_mlp import mlp
        net = mlp(final_embedding_size=132032, num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1)), use_normalization=True)
    elif model_name == "clf_mlp":
        if hparams.clf_head_version == 'v1':
            from .clf_mlp import mlp
            net = mlp(num_classes=2, num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages)))
        elif hparams.clf_head_version == 'v2':
            from .clf_mlp_v2 import mlp
            net = mlp(num_classes=2, num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages)))
        else:
            raise NotImplementedError
        # x -> (b, 96, 4, 4, 4, t)
    elif model_name == "reg_mlp":
        from .clf_mlp import mlp
        net = mlp(num_classes=1, num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages)))
    else:
        raise NameError(f"{model_name} is a wrong model name")

    return net
