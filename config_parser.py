import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.json"

def get_context(config_filename: str = "config.json") -> dict:
    with open(config_filename,'r') as f:
        config = json.load(f);
    
    return {'FRAME_HEIGHT':           config['video']['height'],
                    'FRAME_WIDTH' :   config['video']['width'],
                    'IPV4':           config['network']['ip'],
                    'PORT':           config['network']['port'],
                    'BUFFER_PATH':    config['ipc']['buffer_path'],
                    'SEMAPHORE_NAME': config['ipc']['semaphore_name'],
                    'FPS':            config['video']['fps']
                }


    
    

