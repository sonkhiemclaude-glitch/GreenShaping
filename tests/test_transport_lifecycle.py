"""Session boundary and completion regressions; no EC2 or model inference."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from greenshaping.shaping.pipeline import DPParams, ShapingConfig
from greenshaping.testbed.transport import (
    WIRE_CONTROL,
    WIRE_DATA,
    ShapedTransport,
    WireDecoder,
    _Channel,
    encode_record,
    wire_frame,
)


def test_envelopes_preserve_control_boundary_across_every_chunk_split():
    partial = encode_record({'type': 'reply', 'generation': 1, 'payload': 'truncated'})[:9]
    control = encode_record({'type': 'start', 'generation': 2}, padded=4096)
    data = encode_record({'type': 'reply', 'generation': 2, 'result': 7})
    expected = [(WIRE_DATA, 1, partial), (WIRE_CONTROL, 2, control), (WIRE_DATA, 2, data)]
    wire = b''.join(wire_frame(*frame) for frame in expected)
    for size in [1, 2, 7, 17, 32, 512, 4096, len(wire)]:
        decoder = WireDecoder()
        got = []
        for offset in range(0, len(wire), size):
            got.extend(decoder.feed(wire[offset:offset + size]))
        assert got == expected


def test_old_partial_record_and_late_tail_cannot_corrupt_new_generation():
    values = []
    ch = _Channel.__new__(_Channel)
    ch.lock = threading.RLock()
    ch.generation = 1
    ch.decoder_generation = -1
    ch.wire_counts = {'sent_control':0,'sent_data':0,'received_control':0,'received_data':0}

    def handle(value):
        values.append(value)
        if value['type'] == 'start':
            ch.generation = value['generation']

    ch.handler = handle
    old = encode_record({'type': 'reply', 'generation': 1, 'result': 100})
    ch._dispatch_frame(WIRE_DATA, 1, old[:8])
    ch._dispatch_frame(WIRE_CONTROL, 2, encode_record({'type': 'start', 'generation': 2}, padded=4096))
    ch._dispatch_frame(WIRE_DATA, 1, old[8:])  # Delayed old tail, after start, not a timed decoder reset.
    ch._dispatch_frame(WIRE_DATA, 2, encode_record({'type': 'reply', 'generation': 2, 'result': 7}))
    assert values == [{'type': 'start', 'generation': 2}, {'type': 'reply', 'generation': 2, 'result': 7}]


def test_wire_rejects_invalid_control_size_and_header():
    with pytest.raises(ValueError):
        wire_frame(WIRE_CONTROL, 1, b'too short')
    with pytest.raises(ValueError):
        WireDecoder().feed(b'bad header bytes!!')


def test_finish_waits_for_both_producers_and_freezes_the_snapshot():
    tr = ShapedTransport.__new__(ShapedTransport)
    tr.cfg = ShapingConfig(dp=DPParams(interval=.05, horizon=.5, delta=4096, volume_cap_bytes=2048))
    tr.epoch = time.time() - .5
    tr.end_epoch = time.time()
    tr.condition = threading.Condition()
    tr.ends = {}
    ch = SimpleNamespace(finished=threading.Event(), volume=0, pending=0, cap_exceeded=False,
                         error=None, logs=[{'tick': i} for i in range(9)], timing_failure=False)
    ch.wire_counts = {'sent_control':0,'sent_data':0,'received_control':0,'received_data':0}
    tr._wire_before = dict(ch.wire_counts)
    tr.channels = [ch]

    def complete():
        time.sleep(.35)  # Beyond the old fixed 250ms grace; completion must not be guessed.
        ch.logs.append({'tick': 9})
        ch.finished.set()
        with tr.condition:
            tr.ends[0] = {'timing_valid': True, 'error': None, 'cap_exceeded': False, 'overflow_bytes': 0}
            tr.condition.notify_all()

    thread = threading.Thread(target=complete)
    thread.start()
    stats = tr.finish()
    thread.join()
    assert len(stats['upstream'][0]['ticks']) == 10
    assert stats['completion_ok'] and stats['measurement_valid']
    ch.logs.append({'tick': 100})
    assert len(stats['upstream'][0]['ticks']) == 10


def test_monotonic_schedule_survives_wall_clock_jump_and_marks_late_io(monkeypatch):
    from collections import deque

    import greenshaping.testbed.transport as module

    clock = {'mono': 0., 'wall_offset': 1000., 'sleeps': []}

    def sleep(dt):
        clock['sleeps'].append(dt)
        clock['mono'] += dt
        if len(clock['sleeps']) == 1:
            clock['wall_offset'] += 3600.  # NTP/manual wall-time correction after scheduling begins.

    monkeypatch.setattr(module.time, 'time', lambda: clock['mono'] + clock['wall_offset'])
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock['mono'])
    monkeypatch.setattr(module.time, 'sleep', sleep)
    monkeypatch.setattr(module.np.random, 'default_rng', lambda: SimpleNamespace(laplace=lambda *_: 0.))
    ch = _Channel.__new__(_Channel)
    ch.cfg = ShapingConfig(dp=DPParams(mode='arrivals', interval=.05, horizon=.15, delta=4096,
                                      volume_cap_bytes=2048, bias=10))
    ch.lock = threading.RLock()
    ch.queue = deque()
    ch.volume = ch.pending = 0
    ch.logs = []
    ch.error = None
    ch.cap_exceeded = ch.timing_failure = False
    ch.active = True
    ch.generation = 1
    statuses = []
    ch.send_control = lambda value, completion: (statuses.append(value), completion.set())
    ch._send_wire = lambda *_args, **_kw: None
    ch._ticks(1000., 1, threading.Event())
    assert len(ch.logs) == 3
    assert all(abs(t - .05) < 1e-6 for t in clock['sleeps'])
    assert statuses[0]['timing_valid']

    def delayed(*_args, **_kw):
        clock['mono'] += .1

    clock.update(mono=0., wall_offset=1000., sleeps=[])
    ch.logs = []
    ch.timing_failure = False
    ch._send_wire = delayed
    ch._ticks(1000., 1, threading.Event())
    assert ch.timing_failure
    assert not statuses[-1]['timing_valid']
    assert len(ch.logs) < 3  # No post-horizon catch-up burst to drain stale work.
