"""Functional builder for the multi-branch tabular GAN generator model."""

from tensorflow.keras import layers, Model


ACTIVATION_REGISTRY = {
    "leaky_relu": layers.LeakyReLU,
    "prelu": layers.PReLU,
    "elu": layers.ELU,
    "thresholded_relu": layers.ThresholdedReLU,
}


def build_generator(
        config: dict,
        latent_dim: int,
        continuous_dim: int = 0,
        binary_dim: int = 0,
        discrete_count_dim: int = 0,
        categorical_dims: list[int] | None = None,
) -> Model:
    """Builds a configurable multi-branch generator that maps latent noise to mixed-type tabular data.

    Each output branch uses an activation matched to its feature type: ``sigmoid`` for binary and
    discrete counts, ``tanh`` for continuous, and ``softmax`` per categorical group.

    Args:
        config: Generator config dict. Recognized keys: ``units`` (list of ints, default
            ``[256, 256]``), ``dropout`` (float, default ``0.0``), ``activation`` (str,
            default ``"relu"``), ``batch_norm`` (bool, default ``True``), ``negative_slope`` (float, default ``0.2``).
        latent_dim: Dimensionality of the input noise vector.
        continuous_dim: Number of continuous output features.
        binary_dim: Number of binary output features.
        discrete_count_dim: Number of non-negative integer output features.
        categorical_dims: Per-categorical-feature cardinality list; one softmax branch is built per entry.

    Returns:
        Keras ``Model`` accepting a ``(batch, latent_dim)`` noise tensor and producing a
        concatenated ``(batch, total_dim)`` output vector.

    Raises:
        ValueError: If no output dimensions are configured.
    """
    cat_dims = categorical_dims or []

    units_list = config.get("units", [128, 128])
    dropout_val = config.get("dropout", 0.0)
    activation_name = config.get("activation", "relu").lower()
    use_batch_norm = config.get("batch_norm", True)

    activation_kwargs = {}
    if activation_name == "leaky_relu":
        activation_kwargs["alpha"] = config.get("negative_slope", 0.2)

    noise_input = layers.Input(shape=(latent_dim,))
    x = noise_input

    # Build hidden layers dynamically
    for units in units_list:
        block_input = x

        x = layers.Dense(units)(x)

        if use_batch_norm:
            x = layers.BatchNormalization()(x)

        if activation_name in ACTIVATION_REGISTRY:
            activation_class = ACTIVATION_REGISTRY[activation_name]
            x = activation_class(**activation_kwargs)(x)
        else:
            x = layers.Activation(activation_name)(x)

        if isinstance(dropout_val, float) and dropout_val > 0:
            x = layers.Dropout(dropout_val)(x)

    outputs = []

    # Branch order must match the column order defined by TabularPreprocessor.get_feature_names_out
    if binary_dim > 0:
        outputs.append(layers.Dense(binary_dim, activation="sigmoid", name="binary_out")(x))

    if continuous_dim > 0:
        # tanh bounds continuous outputs to [-1, 1] to match MinMaxScaler(feature_range=(-1, 1))
        outputs.append(layers.Dense(continuous_dim, activation="tanh", name="continuous_out")(x))

    if discrete_count_dim > 0:
        # sigmoid bounds discrete-count outputs to [0, 1] to match MinMaxScaler(feature_range=(0, 1))
        outputs.append(layers.Dense(discrete_count_dim, activation="sigmoid", name="discrete_count_out")(x))

    for i, dim in enumerate(cat_dims):
        outputs.append(layers.Dense(dim, activation="softmax", name=f"categorical_out_{i}")(x))

    # Concatenate all branches into a single output row
    if len(outputs) > 1:
        final_output = layers.Concatenate(axis=-1)(outputs)
    elif len(outputs) == 1:
        final_output = outputs[0]
    else:
        raise ValueError("Generator must have configured output dimensions.")

    return Model(inputs=noise_input, outputs=final_output, name="generator")