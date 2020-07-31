import unittest
from pathlib import Path
import shutil
from src.checkpoint_manager import CheckpointManager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

class TestCheckpointManager(unittest.TestCase):
    def test_save_and_load(self):
        config = {
            'train': {'dir': 'train'}
        }
        p = Path(config['train']['dir'])
        cm = CheckpointManager(p)

        m = Model()
        n_epoch = 0
        optimizer = optim.SGD(m.parameters(), lr=0.01, momentum=0.9)    
        cm.save(m, optimizer, n_epoch)

        m1 = Model()
        last_epoch, step = cm.load(m1, optimizer, cm.latest())

        m_state_dict = {}
        for key, value in m.state_dict().items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            m_state_dict[key] = value

        m1_state_dict = {}
        for key, value in m.state_dict().items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            m1_state_dict[key] = value

        self.assertEqual(m_state_dict.keys(), m1_state_dict.keys())
        for k in m_state_dict.keys():
            self.assertTrue(torch.all(torch.eq(m_state_dict[k], m1_state_dict[k])))

        self.assertEqual(last_epoch, n_epoch)
        self.assertEqual(step, 0)

        shutil.rmtree(p)

    def test_keep_last_n(self):
        config = {
            'train': {'dir': 'train'}
        }
        p = Path(config['train']['dir'])
        cm = CheckpointManager(p)

        m = Model()
        optimizer = optim.SGD(m.parameters(), lr=0.01, momentum=0.9)    

        n_keep = 3
        cm.save(m, optimizer, 0)
        cm.save(m, optimizer, 1)
        cm.save(m, optimizer, 2)
        cm.save(m, optimizer, 3)
        cm.save(m, optimizer, 4, keep=n_keep)

        self.assertEqual(len(cm.list_cp()), n_keep)

        shutil.rmtree(p)
        

if __name__ == '__main__':
    unittest.main()
