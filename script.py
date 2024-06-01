import hydra
from omegaconf import DictConfig, OmegaConf
import subprocess

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    # config = OmegaConf.to_yaml(cfg)
    print("config: ", cfg)
    print("config[main]: ", cfg['main'])
    
    subprocess.run(["python3", "run_main.py", "--cfg", str(cfg)])

if __name__ == "__main__":
    my_app()