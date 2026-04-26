"""
Tasks package — procedurally generated misinformation investigation scenarios.
v2.0: Added PolitifactTask (LIAR dataset), ImageForensicsTask, SECFraudTask.
"""
from env.tasks.task_base import BaseTask
from env.tasks.task_fabricated_stats import FabricatedStatsTask
from env.tasks.task_out_of_context import OutOfContextTask
from env.tasks.task_coordinated_campaign import CoordinatedCampaignTask
from env.tasks.task_politifact import PolitifactTask
from env.tasks.task_image_forensics import ImageForensicsTask
from env.tasks.task_sec_fraud import SECFraudTask
from env.tasks.task_verified_fact import VerifiedFactTask
from env.tasks.task_satire_news import SatiricalClaimTask
from env.tasks.task_plandemic import PlandemicTask

TASK_REGISTRY = {
    "fabricated_stats": FabricatedStatsTask,
    "out_of_context": OutOfContextTask,
    "coordinated_campaign": CoordinatedCampaignTask,
    "politifact_liar": PolitifactTask,
    "image_forensics": ImageForensicsTask,
    "sec_fraud": SECFraudTask,
    "verified_fact": VerifiedFactTask,
    "satire_news": SatiricalClaimTask,
    "plandemic": PlandemicTask,
}

__all__ = [
    "BaseTask",
    "FabricatedStatsTask",
    "OutOfContextTask",
    "CoordinatedCampaignTask",
    "PolitifactTask",
    "ImageForensicsTask",
    "SECFraudTask",
    "VerifiedFactTask",
    "SatiricalClaimTask",
    "PlandemicTask",
    "TASK_REGISTRY",
]
