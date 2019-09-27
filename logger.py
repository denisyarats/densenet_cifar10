from torch.utils.tensorboard import SummaryWriter
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
        ],
        'test': [
            ('step', 'T', 'int'), ('epoch', 'E', 'int'), ('loss', 'L', 'float'),
            ('accuracy', 'A', 'float')
        ],
        'meta': [
            ('step', 'T', 'int'), ('epoch', 'E', 'int'), ('duration', 'D', 'time'),
            ('train_loss', 'TL', 'float'), ('test_loss', 'VL', 'float')
        ],
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
            elif key.startswith('test'):
                key = key[len('test') + 1:]
            else:
                key = key[len('meta') + 1:]
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
        if prefix == 'train':
            color = 'yellow'
        elif prefix == 'test':
            color = 'green'
        else:
            color = 'blue'
        prefix = colored(prefix, color)
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
        if use_tb:
            tb_dir = os.path.join(log_dir, 'tb')
            if os.path.exists(tb_dir):
                shutil.rmtree(tb_dir)
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None
        self._train_mg = MetersGroup(
            os.path.join(log_dir, 'train.log'),
            formating=FORMAT_CONFIG[config]['train'])
        self._test_mg = MetersGroup(
            os.path.join(log_dir, 'test.log'),
            formating=FORMAT_CONFIG[config]['test'])
        self._meta_mg = MetersGroup(
            os.path.join(log_dir, 'meta.log'),
            formating=FORMAT_CONFIG[config]['meta'])

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def _try_sw_log_image(self, key, image, step):
        if self._sw is not None:
            assert image.dim() == 3
            grid = torchvision.utils.make_grid(image.unsqueeze(1))
            self._sw.add_image(key, grid, step)

    def _try_sw_log_video(self, key, frames, step):
        if self._sw is not None:
            frames = torch.from_numpy(np.array(frames))
            frames = frames.unsqueeze(0)
            self._sw.add_video(key, frames, step, fps=30)

    def _try_sw_log_histogram(self, key, histogram, step):
        if self._sw is not None:
            self._sw.add_histogram(key, histogram, step)

    def log(self, key, value, step, n=1):
        assert key.startswith('train') or key.startswith('test') or key.startswith('meta')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value / n, step)
        if key.startswith('train'):
            mg = self._train_mg
        elif key.startswith('test'):
            mg = self._test_mg
        else:
            mg = self._meta_mg
        mg.log(key, value, n)

    def log_param(self, key, param, step):
        self.log_histogram(key + '_w', param.weight.data, step)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias'):
            self.log_histogram(key + '_b', param.bias.data, step)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def log_image(self, key, image, step):
        assert key.startswith('train') or key.startswith('test') or key.startswith('meta')
        self._try_sw_log_image(key, image, step)

    def log_video(self, key, frames, step):
        assert key.startswith('train') or key.startswith('test') or key.startswith('meta')
        self._try_sw_log_video(key, frames, step)

    def log_histogram(self, key, histogram, step):
        assert key.startswith('train') or key.startswith('test') or key.startswith('meta')
        self._try_sw_log_histogram(key, histogram, step)

    def dump(self, step):
        self._train_mg.dump(step, 'train')
        self._test_mg.dump(step, 'test')
        self._meta_mg.dump(step, 'meta')
