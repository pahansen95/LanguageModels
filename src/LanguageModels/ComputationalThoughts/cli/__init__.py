"""


"""
from __future__ import annotations
from dataclasses import dataclass
import logging

from DevAgent.Utils.NatLang.embed import EmbeddingInterface
from DevAgent.Utils.NatLang.chat import ChatInterface
from .. import *

logger = logging.getLogger(__name__)

@dataclass
class TextEncoder(Encoder[str]):
  embedding_interface: EmbeddingInterface

  def __call__(self, input: str) -> latent_t:
    (latent := SemanticLatent(self.embedding_interface)).add_semantics(input, 'ExternalStimuli')
    return latent

@dataclass
class TextDecoder(Decoder[str]):
  chat_interface: ChatInterface

  def __call__(self, latent: latent_t) -> str:
    return self.chat_interface.chat(
      { 'role': 'user', 'content': 'Your goal is to decode a Semantic Latent. A Semantic Latent is a hierachical representation of a thought framed within a iterative train of thought.' },
      { 'role': 'assistant', 'content': 'I understand, please provide me the Semantic Latent' },
      { 'role': 'user', 'content': latent.render() },
      { 'role': 'assistant', 'content': 'How should I decode this Semantic Latent?' },
      { 'role': 'user', 'content': 'Decode the Semantic Latent as a new cohesive thought treating the provided Semantic Latent as your own internal thought process. Your response will be used as the input to the next iteration in a train of thought.' },
    )

async def train_of_thought(
  prompt: str,
  embedding_interface: EmbeddingInterface,
  chat_interface: ChatInterface,
) -> str:
  """Iteratively think on a prompt & generate a response"""

  encode = TextEncoder(embedding_interface)
  decode = TextDecoder(chat_interface)

  async for resp in (
    Pipeline('ToT', [
      encode,
      # Internalize(),
      Think('Contemplate', 'Contemplate', chat_interface),
      # Articulate(),
      decode,
    ])(prompt)
  ):
    logger.info(resp)
    print(resp + '\n\n')
    await asyncio.sleep(0)
