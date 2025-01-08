import os
from box.exceptions import BoxValueError
import yaml
from src.cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read yaml file and return a ConfigBox object
    :param path_to_yaml: Path to yaml file
    :return: ConfigBox object
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as e:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """_summary_

    Args:
        path_to_directories (list):list of path of directories
        verbose (bool, optional): _description_. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """_summary_

    Args:
        path (Path): _description_
        data (dict): _description_
    """
    with open(path,"w") as f:
        json.dump(data,f, indent=4)
    logger.info(f"Json file saved: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load jspn files data
    Args:
    path (Path): path to json file
    Returns:
    ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"Json file loaded: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path:Path):
    """save binary file
    Args:
    data (Any): data to be saved as binary
    path (Path): path to binary file
    """
    joblib.dump(value = data, filename = path)
    logger.info(f"binary file saved at : {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data
    Args:
    path(Path): path to binary file
    Returns:
    Any: object stored in the local file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from : {path}")
    return data

@ensure_annotations
def get_size(path:Path) -> str:
    """get size in kb
    Args:
    path (Path): path to file
    Returns:
    str: size in kb
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~{size_in_kb} KB"

    def decodeImage(imgstring, filename):
        imgdata = base64.b64decode(imgstring)
        with open(filename,'wb') as f:
            f.write(imgdata)
            f.close()

    def encodeImageIntoBase64(croppedImagePath):
        with open(croppedImagePath, "rb") as f:
            return base64.b64encode(f.read())