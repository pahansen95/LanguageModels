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
  chat_interface: ChatInterface
  embedding_interface: EmbeddingInterface

  def __call__(self, input: str) -> latent_t:
    return SemanticLatent(self.embedding_interface).add_semantics(
      input, 'externalStimuli'
    ).add_semantics(
      self.chat_interface.chat(
        { 'role': 'user', 'content': input },
      ),
      'initialThought'
    )

@dataclass
class TextDecoder(Decoder[str]):
  chat_interface: ChatInterface
  embedding_interface: EmbeddingInterface

  def __call__(self, latent: latent_t) -> str:
    logger.debug(f'TextDecoder is Decoding Latent...\n{latent}')
    return self.chat_interface.chat(
      { 'role': 'user', 'content': 'Your goal is to decode a Semantic Latent. A Semantic Latent is a hierachical representation of semantic information rooted at some original semantic.' },
      { 'role': 'assistant', 'content': 'I understand, please provide me the Semantic Latent.' },
      { 'role': 'user', 'content': latent.render() },
      { 'role': 'assistant', 'content': 'How should I decode this Semantic Latent?' },
      { 'role': 'user', 'content': 'Iterate on the original semantic root. Use the embedded child elements to guide you. Your response should be cohesive, articulate & comprehensive without any loss in semantic information. Your response should not refer to the Semantic Latent itself nor include any element tags. Speak as if you are responding directly to a user prompt.' },
    )

async def train_of_thought(
  prompt: str,
  embedding_interface: EmbeddingInterface,
  chat_interface: ChatInterface,
) -> str:
  """Iteratively think on a prompt & generate a response"""

  encode = TextEncoder(chat_interface, embedding_interface)
  decode = TextDecoder(chat_interface, embedding_interface)

  break_pipeline = max_cycles_factory(1)
  idx = 1
  async for resp in (
    Pipeline('ToT', [
      encode,
      Internalize(chat_interface=chat_interface),
      Articulate(chat_interface=chat_interface),
      decode,
    ], break_pipeline=break_pipeline)(prompt)
  ):
    logger.info(resp)
    print(f'<!-- Response Iteration {idx} -->\n\n' + resp.strip() + '\n\n', flush=True)
    await asyncio.sleep(0)
    idx += 1
