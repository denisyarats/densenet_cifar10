from collections import defaultdict
import json
import os
import shutil
import torch
import torchvision
import numpy as np
from termcolor import colored

FORMAT_CONFIG = {
    'sv': {
        'train': [
            ('step', 'T', 'int'), ('epoch', 'E', 'int'), ('duration', 'D', 'time'),
            ('loss', 'L', 'float'), ('accuracy', 'A', 'float'),
            ('learning_rate', 'LR', 'float')
        ],
        'test': [
            ('step', 'T', 'int'), ('epoch', 'E', 'int'), ('loss', 'L', 'float'),
            ('accuracy', 'A', 'float')
        ]
    }
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._file_name = file_name
        if os.path.exists(file_name):
            os.remove(file_name)
        self._formating = formating
        self._meters = defaultdict(AverageMeter)

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('test') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_file(self, data):
        with open(self._file_name, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _format(self, key, value, ty):
        template = '%s: '
        if ty == 'int':
            template += '%d'
        elif ty == 'float':
            template += '%.04f'
        elif ty == 'time':
            template += '%.01f s'
        else:
            raise 'invalid format type: %s' % ty
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = ['{:5}'.format(prefix)]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print('| %s' % (' | '.join(pieces)))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['step'] = step
        self._dump_to_file(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, use_tb=True, config='sv'):
        self._log_dir = log_dir
        self._train_mg = MetersGroup(
            os.path.join(log_dir, 'train.log'),
            formating=FORMAT_CONFIG[config]['train']
        )
        self._test_mg = MetersGroup(
            os.path.join(log_dir, 'test.log'),
            formating=FORMAT_CONFIG[config]['test']
        )

    def log(self, key, value, n=1):
        assert key.startswith('train') or key.startswith('test')
        if type(value) == torch.Tensor:
            value = value.item()
        mg = self._train_mg if key.startswith('train') else self._test_mg
        mg.log(key, value, n)

    def dump(self, step):
        self._train_mg.dump(step, 'train')
        self._test_mg.dump(step, 'test')
