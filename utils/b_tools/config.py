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


if __name__ == "__main__":
    # 读取配置文件并实例化类
    config = yaml2namespace('config/stage_1_sample.yaml')
    print(config.basic_settings.lr == 0.0001)
    print(config.loss_settings.type=="WCE")
    callback = "ModelCheckpoint"
    es = ModelCheckpoint(**vars((getattr(config.callbacks, callback))))
    print(es.filename)

    # print(convert_path("utils/loadData/asvspoof_data_DA.py"))

    # config = load_yaml_config('config/stage_1_sample.yaml')
    # print(config["basic_settings"]["gpuid"])
