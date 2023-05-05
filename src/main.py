import hydra
from omegaconf import DictConfig
import os
import logging
from multi_agent_trainer import MultiAgentTrainer

# from trainer import Trainer
# from distributed_trainer import DistributedTrainer


@hydra.main(config_path="config", config_name="multi_agent_trainer")
def multi_agent_main(cfg: DictConfig, force_resume):
    print(f"Output directory: {os.getcwd()}")
    cfg.common.resume = cfg.common.resume or force_resume
    # logging.getLogger().setLevel(logging.WARNING)
    trainer = MultiAgentTrainer(cfg)
    trainer.run()

def _multi_agent_main(carla_instance, force_resume):
    multi_agent_main()

if __name__ == "__main__":
    multi_agent_main()
