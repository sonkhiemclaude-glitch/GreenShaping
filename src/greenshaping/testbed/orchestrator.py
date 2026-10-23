"""Orchestrator interface (Sec. 6.1).

* `Orchestrator` protocol: run one analyst question against the sites, return the
  per-site matching rows (ground truth for the manifest).
* `ScriptedOrchestrator`: replays the question bank's `plan` over HTTP. It is NOT the
  paper's LLM agent; it exists to validate the capture -> manifest -> attack pipeline
  on the Mac Studio before the LLM orchestrator is connected.
* `LangGraphOrchestrator`: online pilot implementation. The paper uses a LangGraph agent with an
  open-weight local model and a hosted model; the pilot prompts and model metadata are recorded explicitly.
"""
from __future__ import annotations

import http.client
import json
import ssl
from typing import Any, Protocol, TypedDict

from .question_bank import Question


class Orchestrator(Protocol):
    def run(self, question: Question, sites: list[tuple[str, int]]) -> dict[int, int]: ...


class ScriptedOrchestrator:
    """Executes `question.plan` (rounds of calls) over persistent HTTP(S) connections."""

    llm = "none (scripted plan; pipeline validation only)"

    def __init__(self, use_tls: bool = False, cafile: str | None = None, timeout: float = 30.0) -> None:
        self.use_tls, self.cafile, self.timeout = use_tls, cafile, timeout

    def _conn(self, host: str, port: int) -> http.client.HTTPConnection:
        if self.use_tls:
            ctx = ssl.create_default_context(cafile=self.cafile)
            return http.client.HTTPSConnection(host, port, timeout=self.timeout, context=ctx)
        return http.client.HTTPConnection(host, port, timeout=self.timeout)

    def run(self, question: Question, sites: list[tuple[str, int]]) -> dict[int, int]:
        conns = {i: self._conn(h, p) for i, (h, p) in enumerate(sites)}
        rows: dict[int, int] = {}
        try:
            for rnd in question.plan:
                for call in rnd:
                    i = int(call["site"])
                    body = json.dumps({"family": question.family, "params": call["params"]})
                    conns[i].request("POST", "/tool", body=body, headers={"Content-Type": "application/json"})
                    resp = json.loads(conns[i].getresponse().read())
                    if "error" in resp:
                        raise RuntimeError(f"site {i}: {resp['error']}")
                    rows[i] = max(rows.get(i, 0), int(resp["matching_rows"]))
        finally:
            for c in conns.values():
                c.close()
        return rows


class LangGraphOrchestrator:
    """Online local-LLM planning, validated tools and a post-tool LLM summary.

    Explicit pilot design: the analyst supplies typed tool arguments, the LLM
    selects sites, tools execute live, then actual outputs are summarized. No saved
    plan is substituted for a failed model response. This is not a reconstruction
    of unpublished prompts or the paper's unspecified agent implementation.
    """

    def __init__(self, model: str, transport: object, evidence_dir: str,
                 endpoint: str = "http://127.0.0.1:11434", seed: int = 20261021,
                 summarize: bool = True) -> None:
        from pathlib import Path

        from langgraph.graph import END, START, StateGraph

        class State(TypedDict, total=False):
            question: Question
            sites: list[int]
            rows: dict[int, int]
            outputs: list[dict[str, Any]]
            summary: str

        self.model, self.transport, self.endpoint, self.seed = model, transport, endpoint.rstrip("/"), seed
        self.llm = f"Ollama/{model}; online LangGraph; seed={seed}; temperature=0"
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.summarize = summarize
        self.evidence: list[dict[str, Any]] = []
        self.last_window: tuple[float, float] | None = None
        self.last_rows: dict[int, int] = {}
        self.last_transport: dict[str, Any] = {}
        self.last_rounds: list[dict[str, Any]] = []

        def plan(state: State) -> dict[str, Any]:
            q = state['question']
            prompt = ('Select the site IDs explicitly requested by the analyst. '
                      'Output only JSON {"sites":[integer site IDs]}, sorted without duplicates. '
                      'Each request.site is one site ID. ' + q.text)
            expected = sorted({int(c['site']) for rnd in q.plan for c in rnd})
            for attempt in range(3):
                raw = self._chat(prompt, json_mode=True)
                try:
                    chosen = json.loads(raw)['sites']
                    if not isinstance(chosen, list) or any(type(x) is not int for x in chosen) or chosen != expected:
                        raise ValueError('model did not select exactly the analyst-requested sites')
                    return {'sites': chosen}
                except (ValueError, KeyError, TypeError):
                    if attempt == 2:
                        raise RuntimeError('LLM site selection failed after three recorded attempts') from None
                    prompt += ' Recheck every request.site in the analyst input; previous output was invalid.'
            raise AssertionError('unreachable')

        def execute(state: State) -> dict[str, Any]:
            q = state['question']
            rows: dict[int, int] = {}
            outputs = []
            self.last_rounds = []
            self.transport.begin(called_sites=state['sites'])  # type: ignore[attr-defined]
            try:
                pending = list(q.plan)
                while pending:
                    rnd = pending.pop(0)
                    calls = [c for c in rnd if c['site'] in state['sites']]
                    responses = self.transport.call_round(q.family, calls)  # type: ignore[attr-defined]
                    for call, response in zip(calls, responses, strict=True):
                        rows[call['site']] = max(rows.get(call['site'], 0), int(response['matching_rows']))
                    outputs.extend(responses)
                    self.last_rounds.append({'calls': calls, 'responses': responses})
                    if 'adaptive_policy' in q.metadata:
                        from .workload_v4 import next_calls

                        following = next_calls(q, calls, responses, len(self.last_rounds))
                        pending = [following] if following else []
            finally:
                self.last_transport = self.transport.finish()  # type: ignore[attr-defined]
                self.last_window = (self.last_transport['epoch'], self.last_transport['end_epoch'])
                self.last_rows = rows
            return {'rows': rows, 'outputs': outputs}

        def finish(state: State) -> dict[str, Any]:
            if not self.summarize:
                return {'summary': 'Tool outputs retained; LLM summary disabled by configuration.'}
            return {'summary': self._chat('Summarize these synthetic analytics tool outputs in at most two sentences. '
                                          'Do not invent additional data: ' + json.dumps(state['outputs']), json_mode=False)}

        graph = StateGraph(State)
        graph.add_node('plan', plan)
        graph.add_node('execute', execute)
        graph.add_node('summarize', finish)
        graph.add_edge(START, 'plan')
        graph.add_edge('plan', 'execute')
        graph.add_edge('execute', 'summarize')
        graph.add_edge('summarize', END)
        self.graph = graph.compile()

    def _chat(self, prompt: str, json_mode: bool) -> str:
        import time
        import urllib.request

        body: dict[str, object] = {'model': self.model, 'stream': False,
                                  'messages': [{'role': 'user', 'content': prompt}],
                                  'options': {'temperature': 0, 'seed': self.seed, 'num_predict': 128,
                                              'num_ctx': 4096}}
        if json_mode:
            body['format'] = {'type': 'object', 'properties': {'sites': {'type': 'array', 'items': {'type': 'integer'}}},
                              'required': ['sites'], 'additionalProperties': False}
        start = time.time()
        req = urllib.request.Request(self.endpoint + '/api/chat', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.loads(response.read())
        self.evidence.append({'request': body, 'response': result, 'start_epoch': start, 'end_epoch': time.time()})
        return str(result['message']['content'])

    def run(self, question: Question, sites: list[tuple[str, int]]) -> dict[int, int]:
        from ..utils.io import write_json

        self.evidence = []
        self.last_window = None
        try:
            state = self.graph.invoke({'question': question})
            return dict(state['rows'])
        finally:
            write_json({'question_id': question.question_id, 'calls': self.evidence,
                        'transport': self.last_transport, 'window': self.last_window, 'rounds': self.last_rounds},
                       self.evidence_dir / f'q{question.question_id}_{self.transport.generation}.json')  # type: ignore[attr-defined]
