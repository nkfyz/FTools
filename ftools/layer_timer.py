# 统计大模型中每层耗时
from functools import partial

# === Timer for calculate each layer's latency ===
layer_latency = {}

def latency_tik(layer_name, module, input):
    layer_latency[layer_name] = time.time() 

def latency_tok(layer_name,module, input, output):
    layer_latency[layer_name] =  (time.time() - layer_latency[layer_name]) * 1000 # s * 1000 -> ms

model = ...

for layer in model.modules():
  layer.register_forward_pre_hook( partial(latency_tik, layer) )
  layer.register_forward_hook( partial(latency_tok, layer) )
