from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from model import text_audio_sentiment_classify
from train_args import parse_args

args = parse_args()

model = text_audio_sentiment_classify(args)

estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)



# Estimated memory needed for params, optim states and gradients for a:
# HW: Setup with 1 node, 1 GPU per node.
# SW: Model with 7752M total params, 379M largest layer params.
#   per CPU  |  per GPU |   Options
#   194.95GB |   1.41GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
#   194.95GB |   1.41GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
#   173.29GB |  15.85GB | offload_param=none, offload_optimizer=cpu , zero_init=1
#   173.29GB |  15.85GB | offload_param=none, offload_optimizer=cpu , zero_init=0
#     2.12GB | 131.38GB | offload_param=none, offload_optimizer=none, zero_init=1
#    43.32GB | 131.38GB | offload_param=none, offload_optimizer=none, zero_init=0