
import os
import enum
import torch
import numpy as np

from built.trainer import Trainer
from built.builder import Builder
from built.checkpoint_manager import CheckpointManager

class EnsemblePolicy(enum.Enum):
    Even = 0
    HighPriority = 1
    

class Ensembler(object):
    def __init__(self, conf, builder):
        self.config = conf
        self.builder = builder
        self.trainer = Trainer(conf, self.builder)
        
    def exist_model(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            for f in os.listdir(checkpoint_path):
                file_path = os.path.join(checkpoint_path, f)
                _, file_ext = os.path.splitext(file_path)
                if file_ext == '.pth':
                    return True
        
        return False
                

    def forward_models(self, ensemble_policy=EnsemblePolicy.Even):
        
        working_dir = self.config.train.dir

        results = []
        
        for f in os.listdir(working_dir):
            cur_path = os.path.join(working_dir, f)
            if os.path.isdir(cur_path):
                checkpoint_path = os.path.join(cur_path, 'checkpoint')
                if self.exist_model(checkpoint_path):                    
                    cm = CheckpointManager(cur_path)
                    ckpt = cm.latest()
                    last_epoch, step, last_accuracy = cm.load(self.trainer.model, self.trainer.optimizer, ckpt)
                    print(f'{cur_path}:{last_epoch} , {last_accuracy}')

                    output = self.trainer.forward()
                    
                    cur_result = {'name': f, 'output': output, 'accuracy': last_accuracy}
                    
                    results.append(cur_result)
        
        n = len(results)
        print(f'The number of models to ensemble: {n}')
        weights = self.get_weights(results, ensemble_policy)
        
        ensembled = None
        for cur, w in zip(results, weights):
            output = torch.sigmoid(
                cur['output']).cpu().detach().numpy().tolist() * w
            
            if ensembled is None:
                ensembled = output
            else:
                ensembled += output
        
        ensembled = np.argmax(ensembled, axis=1)

            
    def get_weights(self, results, ensemble_policy):
        if ensemble_policy == EnsemblePolicy.Even:
            n = len(results)
            return [1/n] * n
        else:
            raise RuntimeError(f'Ensembel policy({ensemble_policy}) is not supported.')
                    

