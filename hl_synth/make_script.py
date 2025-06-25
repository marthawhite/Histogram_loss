from itertools import product

Y_freqs = [1, 2, 4, 8]
Y_offsets = [0, 10, 20]

depths = [2, 3, 4]
lrs = [2**x for x in range(-12, 5)]
seeds = range(0, 5)

task_idx = 0

for depth, Y_freq, lr, seed in product(depths, Y_freqs, lrs, seeds):
    # task_idx += 1
    # print(f"python sin_functions.py --model_name l2 --depth {depth} --Y_freq {Y_freq} --Y_offset 0 --lr {lr} --seed {seed} --task_idx {task_idx} ")
    # task_idx += 1
    # print(f"python sin_functions.py --model_name HL-Gauss --depth {depth} --Y_freq {Y_freq}  --Y_offset 0 --lr {lr} --seed {seed} --task_idx {task_idx}")
    task_idx += 1
    print(f"python sin_functions.py --model_name HL-Gauss-Balanced --depth {depth} --Y_freq {Y_freq}  --Y_offset 0 --lr {lr} --seed {seed} --task_idx {task_idx}")

for depth, Y_offset, lr, seed in product(depths, Y_offsets, lrs, seeds):
    # task_idx += 1
    # print(f"python sin_functions.py --model_name l2 --depth {depth} --Y_freq 4 --Y_offset {Y_offset} --lr {lr} --seed {seed} --task_idx {task_idx} ")
    # task_idx += 1
    # print(f"python sin_functions.py --model_name HL-Gauss --depth {depth} --hl_range '[-1.5, 21.5]' --Y_freq 4  --Y_offset {Y_offset} --lr {lr} --seed {seed} --task_idx {task_idx}")
    task_idx += 1
    print(f"python sin_functions.py --model_name HL-Gauss-Balanced --depth {depth} --hl_range '[-1.5, 21.5]' --Y_freq 4  --Y_offset {Y_offset} --lr {lr} --seed {seed} --task_idx {task_idx}")