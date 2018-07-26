from env import FieldEnv
from agent import Agent

# file with field data
data_file = 'data/plant_width_mean_dataset.pkl'
# env = FieldEnv(data_file=data_file)
env = FieldEnv(data_file=None)

# single agent (data dependent noise model)
agent = Agent(env, model_type='gpytorch_GP')
# agent = Agent(env, model_type='sklearn_GP')
agent.run(render=True)
