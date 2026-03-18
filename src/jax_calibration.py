import jax
import jax.numpy as jnp
import numpy as np


def fit_temperature_with_jax(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    steps: int = 300,
    lr: float = 0.05
) -> float:
    logits = jnp.array(val_logits)
    labels = jnp.array(val_labels)

    def nll_loss(log_temp):
        temp = jnp.exp(log_temp) + 1e-6
        scaled_logits = logits / temp
        log_probs = jax.nn.log_softmax(scaled_logits, axis=1)
        one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[1])
        return -(one_hot * log_probs).sum(axis=1).mean()

    grad_fn = jax.grad(nll_loss)
    log_temp = jnp.array(0.0)

    for _ in range(steps):
        grad = grad_fn(log_temp)
        log_temp = log_temp - lr * grad

    temperature = float(jnp.exp(log_temp))
    return max(temperature, 1e-6)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits / temperature