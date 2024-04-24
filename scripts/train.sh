CUDA_VISIBLE_DEVICES="6,7" accelerate launch \
                    --multi_gpu \
                    --num_processes=2 \
                    --num_machines=1 src/model_building/train.py
