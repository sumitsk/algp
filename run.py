from env import FieldEnv
from agent import Agent
from arguments import get_args
from pprint import pprint

args = get_args()
pprint(vars(args))
env = FieldEnv(data_file=args.data_file)

# file with field data
# data_file = 'data/plant_width_mean_dataset.pkl'
# env = FieldEnv(data_file=data_file)

# single agent (data dependent noise model)
agent = Agent(env, args)
agent.run(args.render, args.num_runs)
