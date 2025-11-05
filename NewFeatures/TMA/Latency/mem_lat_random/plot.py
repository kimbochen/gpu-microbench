import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

configs = {
    'h200_ldg': { 'label': 'H200 (ldg)', 'color': '#98FB98' },
	'h200_cpa': { 'label': 'H200 (Async Copy)', 'color': '#3CB371' },
    'h200_tma': { 'label': 'H200 (TMA 1D)', 'color': '#006400' },
	'b200_cpa': { 'label': 'B200 (Async Copy)', 'color': 'dimgray' },
	'b200_tma': { 'label': 'B200 (TMA 1D)', 'color': 'black' },
}
global_config = { 'marker': '.', 'linewidth': 2.5, 'markersize': 5 }

data_dir = Path(sys.argv[1])

fig, ax = plt.subplots(figsize=(20, 5))
plt.rcParams.update({'font.size': 18})

for name, config in configs.items():
    x_vals = []
    y_vals = []

    with open(data_dir / f'{name}.txt') as f:
        _ = f.readline()  # Ignore the first line

        for line in f.readlines():
            vals = line.split()
            n_iters = int(vals[0])
            clk_freq = int(vals[1])    # Hz
            data_size = int(vals[2])   # KiB
            bmk_time = float(vals[3])  # ms
            n_cycles = float(vals[4])

            x_vals.append(data_size)
            y_vals.append(n_cycles)

    ax.plot(x_vals, y_vals, **config, **global_config)

ax.set_xlabel('Chain Data Volume (KiB)')
ax.set_ylabel('Latency (Cycles)')
ax.set_xscale('log', base=2)

ax.set_xticks([16, 128, 256, 6*1024, 20*1024, 40*1024, 128*1024])
formatter = FuncFormatter(lambda x, _: f'{x:d} KiB' if x < 1024 else f'{(x // 1024):d} MiB')
ax.get_xaxis().set_major_formatter(formatter)
fig.autofmt_xdate()

ax.set_ylim([0, 1000])
ax.legend()

fig.tight_layout()
fig.savefig('mem_latency.png')
