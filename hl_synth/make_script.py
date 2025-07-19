from itertools import product


# Y_freqs = [1, 2, 4, 8]
# Y_offsets = [0, 10, 20]
# depths = [2, 3, 4]
# lrs = [2**x for x in range(-12, 5)]
# seeds = range(0, 5)

Y_freqs = [1, 10, 20]
Y_offsets = [0]

depths = [3]
widths = [1024]

lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

seeds = range(5)


task_idx = 0

for depth, width, Y_freq, Y_offset, lr, seed in product(depths, widths, Y_freqs, Y_offsets, lrs, seeds):
    task_idx += 1
    print(f"python sin_functions.py --model_name l2 --depth {depth} --width {width} --Y_freq {Y_freq} --Y_offset {Y_offset} --lr {lr} --seed {seed} --task_idx {task_idx} ")
    task_idx += 1
    print(f"python sin_functions.py --model_name HL-Gauss --depth {depth} --width {width} --Y_freq {Y_freq}  --Y_offset {Y_offset} --lr {lr} --seed {seed} --task_idx {task_idx}")
    # task_idx += 1
    # print(f"python sin_functions.py --model_name HL-Gauss-Balanced --depth {depth} --Y_freq {Y_freq}  --Y_offset {Y_offset} --lr {lr} --seed {seed} --task_idx {task_idx}")

Y_freqs = [10]
Y_offsets = [0, 1, 10]
hl_high = 11.5
for depth, width, Y_freq, Y_offset, lr, seed in product(depths, widths, Y_freqs, Y_offsets, lrs, seeds):
    task_idx += 1
    print(f"python sin_functions.py --model_name l2 --depth {depth} --width {width} --Y_freq {Y_freq} --Y_offset {Y_offset} --lr {lr} --seed {seed} --task_idx {task_idx} ")
    task_idx += 1
    print(f"python sin_functions.py --model_name HL-Gauss --depth {depth} --width {width} --Y_freq {Y_freq}  --Y_offset {Y_offset} --lr {lr} --hl_high {hl_high} --seed {seed} --task_idx {task_idx}")
    # task_idx += 1
    # print(f"python sin_functions.py --model_name HL-Gauss-Balanced --depth {depth} --Y_freq {Y_freq}  --Y_offset {Y_offset} --lr {lr} --seed {seed} --task_idx {task_idx}")