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
  
  @property
  def name(self) -> str: return self._name
  @property
  def ready(self) -> bool: return self._input is not None
  @property
  def break_iteration(self) -> Callable[[int, T], bool] | Callable[[int, T], Coroutine[None, None, bool]]:
    return self._break_iteration

  def __aiter__(self) -> AsyncIterator:
    """Asynchronously iterate the pipeline; NOTE: this does not iterate over steps; it is 1 loop step"""
    self._aiter_cycle = 0 # reset the Cycle Count
    return self

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

  def __init__(self, chat_interface: ChatInterface, **kwargs) -> None:
    super().__init__(
      [
        Contemplate(chat_interface=chat_interface),
        # inject_context, # TODO
        Contemplate(chat_interface=chat_interface),
      ],
      **kwargs
    )

class Articulate(Pipeline[latent_t]):
  """A (Sub)Pipeline to Articulate in response to some latent"""

  def __init__(self, chat_interface: ChatInterface, **kwargs) -> None:
    self._chat_interface = chat_interface
    super().__init__(
      name='Articulate',
      steps=[
        Think(
          "Introspect: How can you prime yourself to articulate & clearly communicate your thoughts?",
          chat_interface,
          name='ArticulateThink',
        ),
        self._articulate,
      ],
      **kwargs
    )

  def _articulate(self, latent: latent_t) -> latent_t:
    resp = self._chat_interface.chat(
      {'role': 'user', 'content': 'Think through strategies for articulating your thoughts & clearly communicating them.'},
      {'role': 'assistant', 'content': 'What will I be communicating?'},
      {'role': 'user', 'content': latent.render()},
      {'role': 'assistant', 'content': 'Is there a framework I can use to build a strategy?'},
      {'role': 'user', 'content': 'Here is a basic framework you can extend: A) identify the audience; B) detail key takeaways; C) prioritize information by relevance; D) develop a three part structural outline. Now proceed to strategize.'},
    )
    latent.add_semantics(resp, 'articulation')
    return latent

class Think(Pipeline[latent_t]):
  """A (SubPipeline) to Think on some latent"""
  
  def __init__(self, cognition: str, chat_interface: ChatInterface, **kwargs) -> None:
    """A Pipeline that implements an internal thought process.

    `cognition` is used as a meta-cognitive strategy for the pipeline.
    """
    self._chat_interface = chat_interface
    super().__init__(
      steps=[
        self._step_factory(cognition=cognition, **step_kwargs)
        for step_kwargs in (
            # { 'kind': 'initialThought', 'prompt': 'What is my immediate initial thought?' },
            { 'kind': 'context', 'prompt': 'What do I understand the context to be?' },
            { 'kind': 'audience', 'prompt': 'Who do I understand the audience to be?' },
            { 'kind': 'framing', 'prompt': 'What is my expectation of the audience\'s Frame of Mind?' },
            # { 'kind': 'framing', 'prompt': 'What is my current Frame of Mind?' },
            # { 'kind': 'experiences', 'prompt': 'What relative past experiences do I recall?' },
            { 'kind': 'personalKnowledge', 'prompt': 'What information or details of relevance can I recall?' },
            { 'kind': 'insight', 'prompt': 'What novel ideas or different perspectives do I have, if any?' },
            # { 'kind': 'planning', 'prompt': 'If I was to articulate a response, what would be the priority of information, from most to least important?' }
        )
      ],
      **kwargs
    )
  
  def _step_factory(self, kind: str, cognition: str, prompt: str) -> Callable[..., Any] | Callable[..., Coroutine[None, None, Any]]:
    def _llm_chat(latent: latent_t) -> latent_t:
      resp = self._chat_interface.chat(
        { 'role': 'user', 'content': 'Think through things.' },
        { 'role': 'assistant', 'content': 'What should I consider?' },
        { 'role': 'user', 'content': latent.render() },
        { 'role': 'assistant', 'content': 'How should I guide my cognition?' },
        { 'role': 'user', 'content': f'<meta kind=cognition>{cognition}</meta>', },
        { 'role': 'assistant', 'content': 'What specifically should I ponder?' },
        { 'role': 'user', 'content': f'<meta kind=ponderance>{prompt}</meta>', },
        { 'role': 'assistant', 'content': 'How should I respond?' },
        { 'role': 'user', 'content': 'Your response should be cohesive, articulate & comprehensive. Provide details & examples and avoid circumlocution; speak plainly & with direct meaning & intent.', },
      )
      logger.debug(f'Think({self.name}) Step Output...\n{resp}')
      # TODO: Strip any leading/trailing Element Tags.
      latent.add_semantics(resp, kind)
      return latent
    return _llm_chat

class Contemplate(Think):
  """A Think Pipeline tuned for Contemplation & Thought Expansion"""
  def __init__(self, **kwargs):
    """A Think Pipeline tuned for Contemplation & Thought Expansion"""
    super().__init__(
      name=f'Contemplate',
      cognition='Be contemplative, your goal is to explore your thoughts as much as possible & consider things from many angles.',
      **kwargs,
    )

async def inject_context(latent: latent_t) -> latent_t: ...