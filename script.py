import hydra
from omegaconf import DictConfig, OmegaConf
import subprocess

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    config = OmegaConf.to_yaml(cfg)
    
    
    subprocess.run(["python3", "run_main.py", "--cfg", config])

if __name__ == "__main__":
    my_app()