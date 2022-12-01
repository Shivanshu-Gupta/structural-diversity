from pathlib import Path
import typer

app = typer.Typer()

@app.command()
def main(
        dataset: str = 'covr', split_type: str = 'iid;template',
        subsample_type: str = 'template;subtree', only_incomplete: bool = False,
        subsample_gen: bool = True, train: bool = True,
        node: str = '', n_gpus: int = 8, n_jobs_per_gpu: int = 1,
        outfile: Path = "slurm_script.sh"):
    cmds = []
    if subsample_gen:
        subsample_cmd = f'python -m data_scripts.subsample_pool main --dataset "{dataset}" --split-type "{split_type}" --split-seed -1 --subsample-type "{subsample_type}"'
        if only_incomplete: subsample_cmd += ' --only-incomplete'
        cmds.append(subsample_cmd)
    if train:
        train_cmd = f'srun python -m experiments.parallel_driver exp-1 final0 --dataset "{dataset}" --split-type "{split_type}" --subsample-type "{subsample_type}" --n-gpus {n_gpus} --n-jobs-per-gpu {n_jobs_per_gpu} --train'
        if only_incomplete: train_cmd += f' --only-incomplete'
        cmds.append(train_cmd)
    cmd = ' && \ \n'.join(cmds)
    dataset2node = {
        'covr': 'ava-s0',
        'atis': 'ava-s1',
        'overnight': 'ava-s3',
        # 'thingtalk': 'ava-s0',
        # 'smcalflow-uncommon': 'ava-s4',
    }
    node = node or dataset2node[dataset]
    script = f"""#!/bin/bash

#SBATCH --job-name={dataset}-diversity
#SBATCH --partition=ava_s.p
#SBATCH --nodelist={node}
#SBATCH --cpus-per-task=32
#SBATCH --gpus={n_gpus}
#SBATCH --mem=150GB
#SBATCH --time=0
#SBATCH --output=outputs/slurm.%j.out
#SBATCH --error=outputs/slurm.%j.err

{cmd}
"""
    print(script)
    with open(outfile, 'w') as f:
        f.write(script)

if __name__ == '__main__':
    app()