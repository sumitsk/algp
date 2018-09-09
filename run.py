from env import FieldEnv
from agent import Agent
from arguments import get_args
from pprint import pprint
import os
import json
import numpy as np 
import torch
import matplotlib.pyplot as plt
import ipdb


if __name__ == '__main__':
    args = get_args()
    args_dict = vars(args)
    pprint(args_dict)
    
    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Save arguments as json file
    with open(os.path.join(args.save_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=True)

    env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype)
    # ground_truth(env, args, args.sensor_std)

    agent = Agent(env, args)
    results = agent.run_ipp(render=args.render, num_runs=args.num_runs, criterion=args.criterion, camera_disabled=args.camera_disabled, adaptive=args.adaptive)
    ipdb.set_trace()

    # keys = [i for i in args_dict]
    # if not args.eval_only:
    #     # make a new workbook if not exists
    #     import datetime
    #     now = datetime.datetime.now()
    #     date = str(now.month)+'/'+str(now.day)
    #     time = str(now.hour)+':'+str(now.minute)
    #     args_dict['date'] = date
    #     args_dict['time'] = time
    #     try:
    #         rb = open_workbook(args.logs_wb)
    #     except Exception:
    #         import xlsxwriter
    #         wb = xlsxwriter.Workbook(args.logs_wb)
    #         sh = wb.add_worksheet()
    #         for i, val in enumerate(keys):
    #             sh.write(0, i, val)
    #         wb.close()
    #         rb = open_workbook(args.logs_wb)
        
    #     rsh = rb.sheets()[0]
    #     row = rsh.nrows
    #     wb = copy(rb)
    #     wsh = wb.get_sheet(0)
    #     for i in range(len(keys)):
    #         k = rsh.cell(0, i).value
    #         wsh.write(row, i, args_dict[k])    

    #     wb.save(args.logs_wb)    
    