# smolvla-so101-training

A Viam custom training script that fine-tunes
[SmolVLA](https://huggingface.co/lerobot/smolvla_base) on demonstrations
recorded from an SO-101 arm, and produces a model that the
`viam-labs:vla:policy` service can run.

It is the training script for Viam's course **SO-101: Training a VLA Policy**.
The course records demonstrations of a mug task, but the script works for any
task recorded the same way.

## What it trains on

The script takes a Viam **sequence dataset**: recordings made with
`viam:capture-control`, one sequence per demonstration. Each sequence needs:

- **The follower arm's `JointPositions`**, five joints, in degrees.
- **The gripper's `DoCommand`**, captured with
  `{"docommand_input": {"get": true}}`.
- **Two cameras' `GetImages`**. By default they are named `overhead-camera`
  and `wrist-camera`.
- **A `cmd:` tag** holding the instruction, for example
  `cmd:push the mug to the blue square`.

Every sequence must be recorded at the same capture rate. The policy runs at
that rate, so it is also the rate its controller needs to use.

## What a job does

1. **Copies the images** from the organization's data storage to local disk,
   with the Cloud Storage transfer manager. This is much faster than reading
   them through the job's `/gcs` mount.
2. **Joins the gripper to the arm.** Each arm reading is matched with the
   nearest gripper reading, within 0.05 s. The gripper value is scaled to
   0 = open, 1 = closed, and added as a sixth joint.
3. **Checks the capture rate** is the same in every sequence.
4. **Converts to a LeRobot dataset** with
   [`viam-sequence-to-lerobot`](https://github.com/viam-labs/viam-sequence-to-lerobot).
   The action is the next position, one episode is made per sequence, and the
   instruction comes from the `cmd:` tag.
5. **Checks the dataset**: state and action are six values, both cameras are
   present, and the frame rate and episode count are right.
6. **Fine-tunes `lerobot/smolvla_base`** with `lerobot-train`. The job stops
   if PyTorch cannot use a GPU.
7. **Writes the model**. This is the checkpoint, plus `viam_training.json` and
   `loss_curve.json`, which holds the loss at every logged step.

Every log line is plain `key=value` text tagged `smolvla_so101`. A heartbeat
line every minute says which stage is running, so a quiet log means a stall.

## Running it

The script runs in a training container that already has LeRobot, the
converter and the base models installed. That is
`docker.io/js6450/so101-smolvla:0.4`, registered with Viam as a custom
training container. Built on `huggingface/lerobot-gpu`, it pins LeRobot to the
commit that `viam-labs:vla` 0.2.1 loads checkpoints with.

```bash
viam train submit custom from-registry --dataset-id=<dataset id> --org-id=<org id> --model-name=<model name> --model-version=<new version> --script-name=<script org id>:smolvla-so101 --version=0.6.1 --container-id=<container id> --args=steps=7500
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
| `task_prefix` | `cmd:` | The tag prefix that marks the instruction |
| `clamp_joint` | none | `INDEX:MAX_DEGREES`, for example `3:95`, to clamp one joint |
| `lerobot_args` | none | Any other `lerobot-train` flag, separated by `+`, for example `--policy.optimizer_lr=5e-5+--seed=7` |
| `stage` | `train` | `prepare` stops before training; `probe_io` only tests the GPU and image copy speed |

The training defaults follow `viam-sequence-to-lerobot`'s
`scripts/train_job.toml`.

## Running it locally

`--device mps` or `--device cpu` runs it on a sequence dataset exported with
`viam dataset export`:

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
