# VM.md — PLGrid / Cyfronet Athena (A100) working notes

This file is a “runbook” for working on PLGrid (Cyfronet) Athena from VS Code / SSH, and for running compute-heavy training jobs on A100 GPUs via SLURM.

## Where we are

- **Cluster:** Cyfronet / **Athena**
- **Login node (what you SSH into):** `athena.cyfronet.pl` (often lands on `login01…`)
- **Compute nodes:** scheduled via **SLURM** (hostnames like `t00xx`)
- **Username (PLGrid login):** `plgantoniczapski`
- **GitHub repo example:** `git@github.com:antoniczapski/searchless-chess.git`

### Key rule
Do not run heavy training on the **login node**. Use SLURM (`srun` / `sbatch`) to get compute resources.

---

## Resources and accounting

- **A100 GPU partition:** `plgrid-gpu-a100`
- **Account / grant for billing:** `plgolimpiada2026-gpu-a100`

Check your associations (account/partitions):
```bash
sacctmgr -Pn show assoc user=$USER format=Account,DefaultAccount,Partition
````

See partitions available:

```bash
sinfo -o "%P %a %l %D %N"
```

---

## SSH + VS Code

### Connect from VS Code

Use Remote-SSH to connect to:

```
ssh plgantoniczapski@athena.cyfronet.pl
```

If needed, in `~/.ssh/config` on your local machine (Windows example):

```sshconfig
Host plgrid-athena
  HostName athena.cyfronet.pl
  User plgantoniczapski
  IdentityFile C:\Users\<YOU>\.ssh\plgrid_ed25519
  IdentitiesOnly yes
```

Then in VS Code: `Remote-SSH: Connect to Host…` → `plgrid-athena`.

---

## Typical project layout on Athena

Keep repositories in:

```bash
~/projects/<repo-name>
```

Clone:

```bash
mkdir -p ~/projects
cd ~/projects
git clone git@github.com:antoniczapski/searchless-chess.git
cd searchless-chess
```

---

## Switching from login → compute (interactive session)

### Request an A100 GPU interactively

Run from **login node**:

```bash
srun -p plgrid-gpu-a100 -A plgolimpiada2026-gpu-a100 \
  --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 \
  --pty bash -l
```

You are now on a compute node (prompt changes, `hostname` becomes `t00xx`).

Verify resources:

```bash
hostname
nvidia-smi
```

### Release resources

When finished:

```bash
exit
```

If you don’t exit, your allocation continues until time limit ends, blocking those GPUs for other users.

---

## Persistent Python environment (no conda)

Conda is not available by default (`conda: command not found`). Use a persistent `venv`.

Recommended persistent venv location:

```bash
~/envs/pcg
```

### Create / activate venv

```bash
python3 -m venv ~/envs/pcg
source ~/envs/pcg/bin/activate
python -m pip install -U pip wheel setuptools
```

### Install common deps + GPU PyTorch wheel

Inside the venv:

```bash
pip install numpy pandas matplotlib jupyter nbformat ipykernel tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Sanity check:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

**Important:** install GPU packages while you are on a GPU compute node (after `srun`) to validate CUDA works.

---

## Running notebooks safely (HPC-friendly)

Instead of running Jupyter interactively on login, execute notebooks headlessly on the compute node.

From repo root on the compute node:

```bash
source ~/envs/pcg/bin/activate

python -m jupyter nbconvert \
  --to notebook --execute your_notebook.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output your_notebook.executed.ipynb
```

You can then open `*.executed.ipynb` in VS Code (remote) and inspect outputs.

Download results to local machine (example):

```powershell
scp plgantoniczapski@athena.cyfronet.pl:~/projects/searchless-chess/output_file .
```

---

## Running large training jobs (batch via sbatch)

Interactive is good for debugging. For long runs, use `sbatch`.

### Template: `train.sbatch`

Create a file `train.sbatch` in your repo:

```bash
cat > train.sbatch <<'SBATCH'
#!/bin/bash -l
#SBATCH -J searchless-chess
#SBATCH -p plgrid-gpu-a100
#SBATCH -A plgolimpiada2026-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

source ~/envs/pcg/bin/activate

cd "$SLURM_SUBMIT_DIR"

# Optional: print GPU
nvidia-smi || true

# Run your training
python -u train.py --config configs/run.yaml
SBATCH
```

Submit:

```bash
sbatch train.sbatch
```

Monitor:

```bash
squeue -u $USER
```

Cancel if needed:

```bash
scancel <JOBID>
```

Check historical stats (after completion):

```bash
sacct -j <JOBID> --format=JobID,JobName,Partition,AllocTRES,Elapsed,State,MaxRSS
```

---

## Good cluster hygiene

* Keep training logs in `logs/` and metrics/checkpoints in `runs/` or `checkpoints/`.
* Prefer writing to `$SLURM_SUBMIT_DIR` (your repo working directory) or a dedicated outputs directory in your home.
* Avoid huge writes to shared filesystem in tight loops; buffer logs, write periodically.
* Always `exit` interactive sessions when done.

---

## Quick “am I on login or compute?”

```bash
hostname
```

* login typically contains `login`
* compute typically looks like `t00xx`

---

## Debug checklist

If `srun` fails with “Invalid account or account/partition combination”:

* partition name likely wrong; use:

  ```bash
  sinfo -o "%P"
  ```
* confirm account:

  ```bash
  sacctmgr -Pn show assoc user=$USER format=Account,Partition
  ```

If `torch.cuda.is_available()` is False on compute node:

* confirm `nvidia-smi` works
* ensure you installed torch from `cu121` index URL inside venv
* if cluster uses modules for CUDA, `module avail` / `module load cuda` might be required (site-specific)

---

## Copy/paste quickstart (A100 + run script)

```bash
# on login
cd ~/projects/searchless-chess
srun -p plgrid-gpu-a100 -A plgolimpiada2026-gpu-a100 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash -l

# on compute
source ~/envs/pcg/bin/activate
nvidia-smi
python -u train.py

# done
exit
```
