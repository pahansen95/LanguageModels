"""

Implements the Computational Thought Framework

"""
from __future__ import annotations
from typing import Any, TypeVar, Generic, Protocol
from collections.abc import ByteString, Callable, Coroutine, AsyncIterator
import logging, asyncio, inspect

from .latent import SemanticLatent
from DevAgent.Utils.NatLang.chat import ChatInterface

logger = logging.getLogger(__name__)

### Interfaces & Protocols

T = TypeVar('T')

def max_cycles_factory(max_cycles: int = 1) -> Callable[[int, T], bool]:
  def _max_cycles(idx: int, _: T) -> bool:
    return idx >= max_cycles
  return _max_cycles

class Pipeline(Generic[T]):
  """A Pipeline is an collection of discrete steps that loop until some condition is met"""

  def __init__(self,
    name: str,
    steps: list[Callable[..., Any], Callable[..., Coroutine[None, None, Any]], Pipeline],
    break_iteration: Callable[[int, T], bool] | Callable[[int, T], Coroutine[None, None, bool]] = max_cycles_factory(),
    break_pipeline: Callable[..., bool] | Callable[..., Coroutine[None, None, bool]] = max_cycles_factory(),
  ) -> None:
    self._name = name
    self._input: T | None = None
    self._steps = steps
    self._break_iteration = break_iteration
    self._break_pipeline = break_pipeline
    self._aiter_cycle = 0

  def __aiter__(self) -> AsyncIterator:
    """Asynchronously iterate the pipeline; NOTE: this does not iterate over steps; it is 1 loop step"""
    return self
  
  @property
  def name(self) -> str: return self._name
  @property
  def ready(self) -> bool: return self._input is not None
  @property
  def break_iteration(self) -> Callable[[int, T], bool] | Callable[[int, T], Coroutine[None, None, bool]]:
    return self._break_iteration

  async def __anext__(self) -> T:
    if self._aiter_cycle > 0 and self._break_pipeline(self._aiter_cycle, self._input): raise StopAsyncIteration
    self._input = await self.iterate(self._input, self._break_iteration)
    self._aiter_cycle += 1
    return self._input
  
  async def iterate(self, input: T, break_iteration: Callable[[int, T], bool] | Callable[[int, T], Coroutine[None, None, bool]]) -> T:
    """Execute one iteration of the pipeline; An iteration consists of N Cycles of the steps; The Cycle Break Evaluation function can be overridden."""
    output: T = input
    iter_cycle = 0
    while True:
      iter_cycle += 1
      logger.debug(f'Pipeline {self._name} Cycle {iter_cycle}')
      for step in self._steps:
        if isinstance(step, Pipeline):
          ### Run the Internal Pipeline
          async for step_output in step(output):
            logger.debug(f'Pipline {step.name} Output...\n{step_output}')
            await asyncio.sleep(0)
          output = step_output
        elif inspect.iscoroutinefunction(step): output = await step(output) # Async Function
        else: output = step(output) # Some Synchronous Callable
      if inspect.iscoroutinefunction(break_iteration):
        if (await break_iteration(iter_cycle, output)): break
      else:
        if break_iteration(iter_cycle, output): break
    return output
  
  def __call__(self, input: T) -> Pipeline[T]:
    """Inject the prompt into the pipeline; return the pipeline for chaining"""
    self._input = input
    return self

### Implementations

latent_t = SemanticLatent
"""A Latent Representation of some Semantic Information"""

class Encoder(Generic[T], Protocol):
  """Encodes some arbitrary pipeline input into a Latent"""
  def __call__(self, input: T) -> latent_t: raise NotImplementedError

class Decoder(Generic[T], Protocol):
  """Decodes a Latent into some arbitrary pipeline output"""
  def __call__(self, input: latent_t) -> T: raise NotImplementedError

class Internalize(Pipeline[latent_t]):
  """A (Sub)Pipeline to Internalize some input latent"""

  def __init__(self) -> None:
    super().__init__(
      [
        Think('Contemplate'),
        inject_context,
      ]
    )

class Articulate(Pipeline[latent_t]):
  """A (Sub)Pipeline to Articulate in response to some latent"""

  def __init__(self) -> None:
    super().__init__(
      [
        Think('Articulate'),
      ]
    )

class Think(Pipeline[latent_t]):
  """A (SubPipeline) to Think on some latent"""
  
  def __init__(self, name: str, cognition: str, chat_interface: ChatInterface) -> None:
    """cognition is some description, long or short, instructing how to think."""
    self._chat_interface = chat_interface
    self._step_kwargs = [
        { 'kind': 'context', 'prompt': 'What do I understand the context of my thoughts to be?' },
        { 'kind': 'framing', 'prompt': 'What is my current Frame of Mind?' },
        { 'kind': 'experiences', 'prompt': 'Can I recall any past experiences that relate?' },
        { 'kind': 'personalKnowledge', 'prompt': 'Can I remember any relevant information or details? How confident am I in this knowledge\'s accuracy & precision?' },
        { 'kind': 'insight', 'prompt': 'Do I have any novel ideas or different perspectives?' },
    ]
    super().__init__(
      name,
      [
        self._step_factory(cognition=cognition, **step_kwargs)
        for step_kwargs in self._step_kwargs
      ]
    )
  
  def _step_factory(self, kind: str, cognition: str, prompt: str) -> Callable[..., Any] | Callable[..., Coroutine[None, None, Any]]:
    def _llm_chat(latent: latent_t) -> latent_t:
      resp = self._chat_interface.chat(
        { 'role': 'user', 'content': 'You are actively thinking; this is your inner monologue.' },
        { 'role': 'assistant', 'content': 'What are my current thoughts?' },
        { 'role': 'user', 'content': latent.render() },
        { 'role': 'assistant', 'content': 'What is my active cognition?' },
        { 'role': 'user', 'content': f'<meta kind=cognition>{cognition}</meta>', },
        { 'role': 'assistant', 'content': 'What am I currently pondering?' },
        { 'role': 'user', 'content': f'<meta kind=ponderance>{prompt}</meta>', },
        { 'role': 'assistant', 'content': 'How should I respond?' },
        { 'role': 'user', 'content': 'Embody your inner voice & reply directly to yourself.', },
      )
      logger.debug(f'Think({self.name}) Step Output...\n{resp}')
      # TODO: Strip any leading/trailing Element Tags.
      latent.add_semantics(resp, kind)
      return latent
    return _llm_chat

async def inject_context(latent: latent_t) -> latent_t: ...