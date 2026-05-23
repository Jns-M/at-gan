"""Functional builder for the GAN discriminator model."""

from tensorflow.keras import layers, Model


ACTIVATION_REGISTRY = {
    "leaky_relu": layers.LeakyReLU,
    "prelu": layers.PReLU,
    "elu": layers.ELU,
    "thresholded_relu": layers.ThresholdedReLU,
}


def build_discriminator(input_dim: int, config: dict) -> Model:
    """Builds a configurable dense discriminator that outputs a real/fake probability.

    Args:
        input_dim: Width of the input feature vector (may be ``total_dim * pack_size``).
        config: Discriminator config dict. Recognized keys: ``units`` (list of ints),
            ``dropout`` (float, default ``0.2``), ``activation`` (str, default
            ``"leaky_relu"``), ``negative_slope`` (float, default ``0.2``).

    Returns:
        Compiled Keras ``Model`` with a single sigmoid output named ``validity_out``.
    """
    units_list = config.get("units", [256, 256])
    dropout_val = config.get("dropout", 0.2)
    activation_name = config.get("activation", "leaky_relu").lower()

    activation_kwargs = {}
    if activation_name == "leaky_relu":
        activation_kwargs["alpha"] = config.get("negative_slope", 0.2)

    data_input = layers.Input(shape=(input_dim,))
    x = data_input

    # Build hidden layers dynamically
    for units in units_list:
        x = layers.Dense(units)(x)

        if activation_name in ACTIVATION_REGISTRY:
            activation_class = ACTIVATION_REGISTRY[activation_name]
            x = activation_class(**activation_kwargs)(x)
        else:
            x = layers.Activation(activation_name)(x)

        if dropout_val > 0:
            x = layers.Dropout(dropout_val)(x)

    validity = layers.Dense(1, activation="sigmoid", name="validity_out")(x)

    return Model(inputs=data_input, outputs=validity, name="discriminator")