# global variable cross files

def _init():
    global _global_dict
    _global_dict = {}

def set_value(name, value):
    _global_dict[name] = value

def get(name, defValue=0):
    if name in _global_dict:
        return _global_dict[name]
    else:
        return defValue

# 累加统计数据，如数量，loss，correct
def summary(**kwargs):
    for key in kwargs:
        if key == 'number':
            get('statics')['number'] += kwargs['number']
        else:   
            get('statics')[key] += kwargs['number'] * kwargs[key]

def clear_statics(*args):
    if len(args) == 0:
        for key in get('statics'):
            get('statics')[key] = 0
    else:
        for key in args:
            get('statics')[key] = 0


_init()

set_value('epoch', 1)
set_value('statics', {})
set_value('best_epoch', 0)
set_value('best_accuracy', 0.0)
set_value('best_loss', 1.0)
set_value('log', [])

