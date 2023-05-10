import hydra
from omegaconf import DictConfig
import os
import logging
from multi_agent_trainer import MultiAgentTrainer

# from trainer import Trainer
# from distributed_trainer import DistributedTrainer


def multi_agent_main(force_resume):
    print(f"Output directory: {os.getcwd()}")
    logging.getLogger().setLevel(logging.WARNING)
    
    @hydra.main(config_path="config", config_name="multi_agent_trainer")
    def inner_ma_main(cfg):
        cfg.common.resume = cfg.common.resume or force_resume
        trainer = MultiAgentTrainer(cfg)
        trainer.run()
        
    inner_ma_main()

def _multi_agent_main(carla_instance, force_resume):
    multi_agent_main(force_resume)

if __name__ == "__main__":
    multi_agent_main()
