# Module contains any methods, class, parameters, etc that is related to logging the trainig

import io

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import wandb

def record(item, writer, step):
    summary = tf.Summary()
    for key, value in item.items():
        summary.value.add(tag=key, simple_value=value)
    writer.add_summary(summary, step)
    writer.flush()

def tb_log_histogram(data, tag, step, **kargs):
    # Log histogram via Weights & Biases
    try:
        wandb.log({tag: wandb.Histogram(np.asarray(data))}, step=int(step))
    except Exception:
        pass

def tb_log_ctf_frame(frame, tag, step):
    num_images = frame.shape[2]
    fig = plt.figure(1)
    ncol = 6
    nrow = (num_images//ncol)+1
    scale = 3
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol*scale, nrow*scale))
    for n, ax in zip(range(num_images), axs.ravel()):
        image = frame[:,:,n]
        im = ax.imshow(image)
        ax.set_title('ch {0}'.format(n))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax)

    # Log image via Weights & Biases
    try:
        wandb.log({tag: wandb.Image(fig)}, step=int(step))
    except Exception:
        pass
    plt.close(fig)
