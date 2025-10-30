<div align='center'>
<h1> DISSC: Disentangling Successor Features for Cordination </h1>
</div>

## Important Dependencies

- StarCraft II Multi-Agent Challenge (SMAC)

[SMAC repository](https://github.com/oxwhirl/smac)

- Tensorflow 2.3

## Example usage

``` bash
python run_cvdc.py --machine <machine_name> --silence --training_steps 2000000 --train_number 01 --map 2s3z
```

## W&B logging and sweeps

Training logs are sent to Weights & Biases. To run a sweep, create a sweep file like `wandb_sweeps/cvdc_example.yaml` and use:

``` bash
wandb sweep wandb_sweeps/cvdc_example.yaml
# then
wandb agent <entity>/<project>/<SWEEP_ID>
```

Supported sweep keys (snake_case) are mapped into the internal config: decentral_lr, central_lr, entropy_beta, psi_beta, reward_beta, decoder_beta, critic_beta, q_beta, learnability_beta, target_kl, eps, buffer_size, minibatch_size, epoch, save_model_frequency, save_statistic_frequency, save_image_frequency, moving_average_step, test_interval, test_episode, frame_stack, gamma, lambda_, training_steps, step_mul, difficulty, map.

## Author

Seung Hyun Kim - skim449 (skim449@illinois.edu)
