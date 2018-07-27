from env import FieldEnv
from agent import Agent

env = FieldEnv(data_file=None)

# file with field data
# data_file = 'data/plant_width_mean_dataset.pkl'
# env = FieldEnv(data_file=data_file)

# single agent (data dependent noise model)
agent = Agent(env, model_type='gpytorch_GP', lr=.05)
# agent = Agent(env, model_type='sklearn_GP')
agent.run(render=True)
