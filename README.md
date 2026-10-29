# smolvla-so101-training

A Viam custom training script that fine-tunes
[SmolVLA](https://huggingface.co/lerobot/smolvla_base) on demonstrations
recorded from an SO-101 arm, and produces a model that the
`viam-labs:vla:policy` service can run. It works for any task: the
instruction for each demonstration comes from a tag, and the component names
are arguments.

## What it trains on

The script takes a Viam **sequence dataset**: recordings made with
`viam:capture-control`, one sequence per demonstration. Each sequence needs:

- **The arm's `JointPositions`**, in degrees. The model's state is the arm's
  joints, plus one for the gripper if there is one.
- **The gripper's `DoCommand`**, captured with
  `{"docommand_input": {"get": true}}`. The response needs a numeric
  `position`; set `gripper_open` and `gripper_closed` to its value at each end.
  To train without a gripper, pass `gripper=` empty.
- **Each camera's `GetImages`**. The defaults are an overhead camera and a
  wrist camera.

The component names default to `follower-arm`, `follower-gripper`,
`overhead-camera` and `wrist-camera`. If yours differ, pass `arm`, `gripper`
and `cameras` (see [Arguments](#arguments)).
- **A tag holding the instruction**: `cmd:`, for example
  `cmd:push the mug to the blue square`, or `task:`, which is what the
  [capture-control web app](https://github.com/viam-labs/capture-control-webapp)
  writes. A dataset can mix the two.

Every sequence must be recorded at the same capture rate. The policy runs at
that rate, so it is also the rate its controller needs to use.

## What a job does

1. **Copies the images** from the organization's data storage to local disk,
   with the Cloud Storage transfer manager. This is much faster than reading
   them through the job's `/gcs` mount.
2. **Joins the gripper to the arm.** Each arm reading is matched with the
   nearest gripper reading, within 0.05 s. The gripper value is scaled to
   0 = open, 1 = closed, and added after the arm's joints.
3. **Checks the capture rate** is the same in every sequence. A sequence more
   than 2% off the median, usually one whose first camera dropped frames, is
   left out with a warning and listed in `viam_training.json`; if more than 10%
   are off, the job stops, because the recording rate itself is inconsistent.
4. **Converts to a LeRobot dataset** with
   [`viam-sequence-to-lerobot`](https://github.com/viam-labs/viam-sequence-to-lerobot).
   The action is the next position, one episode is made per sequence, and the
   instruction comes from the `cmd:` tag, or the `task:` tag if there is no
   `cmd:` one.
5. **Checks the dataset**: state and action have one value per joint, every
   camera is present, and the frame rate and episode count are right.
6. **Fine-tunes `lerobot/smolvla_base`** with `lerobot-train`. The job stops
   if PyTorch cannot use a GPU.
7. **Writes the model**. This is the checkpoint, plus `viam_training.json` and
   `loss_curve.json`, which holds the loss at every logged step.

Every log line is plain `key=value` text tagged `smolvla_so101`. A heartbeat
line every minute says which stage is running, so a quiet log means a stall.

## Running it

The script is `smolvla-so101` in the Viam registry. Run it in the public
training container **LeRobot SmolVLA for SO-101 (0.1: NVIDIA driver env)**.
The container already has LeRobot, the converter and the base models
installed. It is built on `huggingface/lerobot-gpu` and pins LeRobot to the
commit that `viam-labs:vla` 0.2.1 loads checkpoints with, so a model you
train with it can be deployed.

In the Viam app, train a model on your sequence dataset with **Custom**
training: choose `smolvla-so101` as the script and that container from the
container list, then add any [arguments](#arguments).

With the CLI (version 1.12.0 or later), find the container's ID with
`viam train containers list --org-id=<org id>`, then submit:

```bash
viam train submit custom from-registry --dataset-id=<dataset id> --org-id=<org id> --model-name=<model name> --model-version=<new version> --script-name=<namespace>:smolvla-so101 --version=<script version> --container-id=<container id> --args=steps=7500
```

```bash
viam train logs --job-id=<job id>
```

Every `--model-version` must be new. To deploy the model, use the
`viam-labs:vla:policy` service, then pin the package's `version` to the one
you trained.

## Arguments

Pass these with `--args key=value,key=value`. Viam splits `--args` on commas,
so a value that holds a list uses `+` between its items.

| Argument | Default | What it sets |
| --- | --- | --- |
| `steps` | 3750 | Training steps. The learning-rate schedule and checkpoint interval scale with it |
| `batch_size` | 64 | Samples per step |
| `n_action_steps` | 30 | Actions the policy plays out per inference, at most 50. Saved in the model's `config.json` |
| `use_amp` | `true` | Mixed precision |
| `num_workers` | 12 | Data loader workers |
| `arm` | `follower-arm` | The arm component |
| `gripper` | `follower-gripper` | The gripper component; empty for no gripper |
| `gripper_open`, `gripper_closed` | 100, 0 | The gripper's raw readings when open and closed |
| `cameras` | `overhead-camera wrist-camera` | Camera names, separated by spaces. The first sets the frame clock |
| `task_prefix` | `cmd: task:` | The tag prefixes that mark the instruction, separated by spaces and tried in order |
| `clamp_joint` | none | `INDEX:MAX_DEGREES`, for example `3:95`, to clamp one joint |
| `lerobot_args` | none | Any other `lerobot-train` flag, separated by `+`, for example `--policy.optimizer_lr=5e-5+--seed=7` |
| `stage` | `train` | `prepare` stops before training; `probe_io` only tests the GPU and image copy speed |

The training defaults follow `viam-sequence-to-lerobot`'s
`scripts/train_job.toml`.

## Running it locally

Install it with `pip install .`, which installs LeRobot and the converter at
the pinned commits (the container uses Python 3.12), and have `ffmpeg` on your
`PATH`. `--device mps` or `--device cpu` then runs it on a sequence dataset
exported with `viam dataset export`:

```bash
python -m model.training --binary_data_file=<dir>/binary_data.parquet --tabular_data_file=<dir>/tabular_data.parquet --sequences_file=<dir>/sequences.parquet --model_output_directory=<out> --device=mps --steps=200
```

This is slow, so it is meant for checking a dataset, not for training a model.

## Publishing a new version

```bash
tar -czf smolvla-so101-<version>.tar.gz setup.py model
```

```bash
viam training-script upload --org-id=<org id> --path=smolvla-so101-<version>.tar.gz --script-name=smolvla-so101 --version=<version> --framework=registry --module-resource=viam-labs:vla:policy --visibility=<private|public>
```

Bump `version` in `setup.py` to match.

## License

Apache 2.0. See `LICENSE`.
