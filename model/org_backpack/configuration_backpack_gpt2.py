from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class BackpackGPT2Config(GPT2Config):
  """
    This is the configuration class to store the configuration of a [`GPT2Model`] or a [`TFGPT2Model`]. It is used to
    instantiate a Backpack GPT-2 model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`GPT2Config`] and can be used to control the model outputs. Read the
    documentation from [`GPT2Config`] for more information.

    Args:
        num_senses (`int`, *optional*, defaults to 16):
            The number of sense vectors to define for each word.
        sense_intermediate_scale (`int`, *optional*, defaults ot 4):
            The hidden dimensionality of the sense vector network.

    Example:

    ```python
    >>> from transformers import BackpackGPT2Config, BackpackGPT2Model

    >>> # Initializing a GPT2 configuration
    >>> configuration = BackpackGPT2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = BackpackGPT2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
  """

  def __init__(self,
               vocab_size=50264,
               num_senses=16,
               sense_intermediate_scale=4,
               n_positions=512,
               scale_attn_by_inverse_layer_idx=True,
               **kwargs,
  ):
    self.num_senses = num_senses
    self.sense_intermediate_scale = sense_intermediate_scale
    super().__init__(vocab_size=vocab_size, n_positions=n_positions, scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx, **kwargs)
