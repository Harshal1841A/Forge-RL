# server/app.py
from openenv.core.env_server import create_app
from forge_environment import ForgeEnvironment
from models import ForgeAction, ForgeObservation

env = ForgeEnvironment()
app = create_app(env, ForgeAction, ForgeObservation)
