"""GAN-inspired adversarial self-play agents for co-evolutionary training."""
from agents.adversarial.generator_agent import GeneratorAgent
from agents.adversarial.self_play import SelfPlayTrainer

__all__ = ["GeneratorAgent", "SelfPlayTrainer"]
