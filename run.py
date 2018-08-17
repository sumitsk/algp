from env import FieldEnv
from agent import Agent
from arguments import get_args
from pprint import pprint
import os
import json


if __name__ == '__main__':
    args = get_args()
    pprint(vars(args))

    # Save arguments as json file
    with open(os.path.join(args.save_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=True)

    env = FieldEnv(data_file=args.data_file)

    # file with field data
    # data_file = 'data/plant_width_mean_dataset.pkl'
    # env = FieldEnv(data_file=data_file)

    # single agent (data dependent noise model)
    agent = Agent(env, args)
    agent.run(args.render, args.num_runs)
