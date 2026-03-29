def tood_build_config():
    return dict(
        backbone=dict(
            type="Unet", 
            n_blocks=[2, 3, 4, 5], 
            norm_type='InstanceNorm', 
            act_type='LeakyReLU',
            coord=True),
        head=dict(
            type='TOOD',
            cls=1,
            in_channels=96,
            reg_max=8,
        ))

def basic_build_config():
    return dict(
        backbone=dict(
            type="Unet", 
            n_blocks=[2, 3, 4, 5], 
            norm_type='InstanceNorm', 
            act_type='LeakyReLU',
            coord=True),
        head=dict(
            type='ClsRegHead',
            cls=1,
            in_channels=96,
            reg_max=8,
            conv_num=2,
        ))
