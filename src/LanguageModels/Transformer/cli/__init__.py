"""

Provides CLI Subcommands for the `Transformer` Sub-Package

"""

import torch, logging
from .. import Transformer

logger = logging.getLogger(__name__)

async def test_transformer(*args, **kwargs) -> int:
  """Tests the Assembly of a Vanilla Transformer Architecture & its operation with random data"""
  logger.info('Running Transformer Tests')

  # Mock parameters
  src_vocab_size = 100
  tgt_vocab_size = 100
  d_model = 64
  num_layers = 2
  num_heads = 4
  d_ff = 256
  max_len = 10
  batch_size = 2

  logger.info(
    "Initializing Transformer Model with Config: "
    f"src_vocab_size={src_vocab_size}, tgt_vocab_size={tgt_vocab_size}, "
    f"d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}, "
    f"d_ff={d_ff}, max_len={max_len}"
  )

  # Initialize Transformer
  transformer = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    max_len=max_len
  )

  logger.info("Creating random input data")
  # Random input data
  src = torch.randint(0, src_vocab_size, (batch_size, max_len))
  tgt = torch.randint(0, tgt_vocab_size, (batch_size, max_len))
  src_mask = torch.ones((batch_size, 1, max_len))
  tgt_mask = torch.ones((batch_size, 1, max_len))

  logger.info(
    f"Input shapes -- src: {src.shape}, tgt: {tgt.shape}, "
    f"src_mask: {src_mask.shape}, tgt_mask: {tgt_mask.shape}"
  )

  logger.info("Performing a forward pass")
  # Perform a forward pass
  output = transformer(src, tgt, src_mask, tgt_mask)

  # Check output shape
  logger.info(f"Output shape: {output.shape}, Expected: {(batch_size, max_len, tgt_vocab_size)}")
  assert output.shape == (batch_size, max_len, tgt_vocab_size), f"Output shape {output.shape} mismatch!"
  
  # Check that the model output is of type torch.Tensor
  logger.info("Checking output type")
  assert isinstance(output, torch.Tensor), "Output is not a torch.Tensor!"

  logger.info("Transformer test passed successfully.")
  return 0
  