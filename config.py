data_config = {
    # directory for putting all preprocessed results for training to this path
    'use_cpm_data' : False,
    'data_dir': "/nvmedata/fengxingyu/preprocessed",
    'use_pn9' : False,
    'pn9_data_dir' : "xx/xx",
    'crop_size': (96, 96, 96),
    'experiment_dir' : 'experiment/',
}

net_config = {
    # Net configuration
    'chanel': 1,
    'crop_size': data_config['crop_size'],
    'infer_topk' : 32
}

loss_config = {
    "ratio" : 80,
    "reg_max" : 8,
    "pos_threshold" : 0.9,
    "cls_weight" : 1.0,
    "offset_weight" : 0.5,
    "shape_weight" : 0.5,
    "class_neg_weight" : 0.1,
} 

train_config = {
    'gpu_id' : "1",
    'fold_num' : 6,
    'batch_size': 2,
    'momentum': 0.9,
    'init_lr' : 1e-3,
    'weight_decay': 1e-4,
    'warm_up': 30,

    'epochs': 60,
    'epoch_save': 5,
    'num_workers': 20,
}

test_config = {
    'ckpt': "tb_logs/topk-7/fold6/my_model/version_0/checkpoints/best.ckpt",
}

config = dict(data_config, **net_config)
config = dict(config, **train_config)
config = dict(config, **loss_config)
config = dict(config, **test_config)

