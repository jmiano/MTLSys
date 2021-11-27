import torch_pruning as tp
from torch import nn
import torch
import time
from utils.evaluation import run_evaluation
import numpy as np


def prune_model(model, PRUNING_PERCENT=0.2):
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))
    strategy = tp.strategy.L1Strategy() 
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            pruning_idxs = strategy(module.weight, amount=PRUNING_PERCENT) # or manually selected pruning_idxs=[2, 6, 9, ...]
            pruning_plan = DG.get_pruning_plan(module, tp.prune_conv, idxs=pruning_idxs )
            pruning_plan.exec()
        if isinstance(module, torch.nn.Linear):
            if 'class0' not in name:
                pruning_idxs = strategy(module.weight, amount=PRUNING_PERCENT) # or manually selected pruning_idxs=[2, 6, 9, ...]
                pruning_plan = DG.get_pruning_plan(module, tp.prune_linear, idxs=pruning_idxs )
                pruning_plan.exec()
            
    return model


def prune_other_tasks(model, task1, task2, PRUNING_PERCENT = 0.1):
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))
    strategy = tp.strategy.L1Strategy() 
    for name, module in model.named_modules():
        if(task1 in name or task2 in name): 
            if isinstance(module, torch.nn.Linear):
                pruning_idxs = strategy(module.weight, amount=1) # or manually selected pruning_idxs=[2, 6, 9, ...]
                pruning_plan = DG.get_pruning_plan(module, tp.prune_linear, idxs=pruning_idxs )
                pruning_plan.exec()
                
            
        if isinstance(module, torch.nn.Conv2d):
            pruning_idxs = strategy(module.weight, amount=PRUNING_PERCENT) # or manually selected pruning_idxs=[2, 6, 9, ...]
            pruning_plan = DG.get_pruning_plan(module, tp.prune_conv, idxs=pruning_idxs )
            pruning_plan.exec()
        if isinstance(module, torch.nn.Linear):
            if 'class0' not in name:
                pruning_idxs = strategy(module.weight, amount=PRUNING_PERCENT) # or manually selected pruning_idxs=[2, 6, 9, ...]
                pruning_plan = DG.get_pruning_plan(module, tp.prune_linear, idxs=pruning_idxs )
                pruning_plan.exec()
            
    return model


def get_f1_and_lat(model_path, eval_dataset, eval_dataloader, tasks, mtl_model=True):
    # Get latency
    latencies = []
    for i, sample in enumerate(eval_dataset):
        start = time.time()
        model = torch.load(f'models/{model_path}')
        model = model.cpu()
        image = sample[0].unsqueeze(0)
        output = model(image)
        lat = time.time() - start
        latencies.append(lat)
        if i >= 100:
            break
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)

    # Get scores
    model = model.cuda()
    scores = run_evaluation(model, eval_dataloader, tasks, mtl_model=mtl_model)

    return [scores, [mean_lat, std_lat]]


