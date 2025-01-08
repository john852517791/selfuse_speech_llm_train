import yaml
import argparse
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint,LearningRateMonitor
CALLBACK_MAP = {
        "early_stop":EarlyStopping,
        "ModelCheckpoint":ModelCheckpoint,
        "LearningRateMonitor":LearningRateMonitor,
    }

def load_yaml_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def dict_to_namespace(d):
    namespace = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def yaml2namespace(ymlpath):
    return dict_to_namespace(load_yaml_config(ymlpath))

def convert_path(original_path):
    return original_path.replace('/', '.').replace('.py', '')

def get_callbacks(conf):
    
    cbs = vars(conf)
    res = []
    # earlystop
    for callback in cbs:
        if callback in CALLBACK_MAP:
            res.append(CALLBACK_MAP[callback](**(getattr(conf, callback))))
        else:
            print("invalid callback config")
    return res



class Config_sample:
    """通用配置类，将字典转为类属性"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):  # 如果是字典，递归转换为 Config 对象
                setattr(self, key, Config_sample(value))
            else:
                setattr(self, key, value)

def load_yaml_to_class(file_path, cls=Config_sample):
    """读取 YAML 文件并转换为指定类"""
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return cls(config_dict)


if __name__ == "__main__":
    # 读取配置文件并实例化类
    # config = yaml2namespace('config/stage_1a.yaml')
    # print(config.basic_settings.lr == 0.0001)
    # print(config.loss_settings.type=="WCE")
    # callback = "ModelCheckpoint"
    # es = ModelCheckpoint(**vars((getattr(config.callbacks, callback))))
    # print(es.filename)
    print(load_yaml_to_class('config/stage_1a.yaml').basic_settings.data_module)

    # print(convert_path("utils/loadData/asvspoof_data_DA.py"))

    # config = load_yaml_config('config/stage_1_sample.yaml')
    # print(config["basic_settings"]["gpuid"])