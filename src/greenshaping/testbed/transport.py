"""Experimental persistent TLS byte-stream transport for a single-host pilot.

The DP-only mode emits fixed ticks in both directions on ALL links, with bounded
input volume, independent Laplace noise, fixed horizon and no overflow flush.
Control frames are fixed-size and sent to all sites before a public start epoch.
This is an engineering pilot, not a verified end-to-end DP implementation: OS/TLS
packetisation and scheduling jitter must be audited independently. Rounds use
padded calls, fixed reply chunks and all-active polling; compositions
remain experimental and require separate wire validation.
"""
from __future__ import annotations

import copy
import json
import math
import queue
import select
import socket
import ssl
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ..shaping.dp_release import laplace_scale
from ..shaping.padding import pad_size
from ..shaping.pipeline import ShapingConfig
from ..shaping.rounds import draw_active_set

MAX_RECORD = 4 * 1024 * 1024
WIRE_HEADER = struct.Struct('!4sBQI')
WIRE_MAGIC = b'GSW1'
WIRE_CONTROL, WIRE_DATA = 0, 1
MAX_WIRE_FRAME = 64 * 1024 * 1024  # Public transport safety limit; exceeding it invalidates the session.


def wire_frame(kind: int, generation: int, payload: bytes) -> bytes:
    if kind not in (WIRE_CONTROL, WIRE_DATA) or not 0 <= generation < 2**64:
        raise ValueError('invalid wire envelope')
    if len(payload) > MAX_WIRE_FRAME or (kind == WIRE_CONTROL and len(payload) != 4096):
        raise ValueError('wire envelope exceeds public transport limit')
    return WIRE_HEADER.pack(WIRE_MAGIC, kind, generation, len(payload)) + payload


class WireDecoder:
    """Separate control and generation-tagged tick fragments before application decoding.

    A truncated application record cannot consume the next session's control frame.
    Envelope bytes are encrypted by TLS and included in measured wire cost.
    """

    def __init__(self) -> None:
        self.buffer = bytearray()

    def feed(self, data: bytes) -> list[tuple[int, int, bytes]]:
        self.buffer.extend(data)
        out = []
        while len(self.buffer) >= WIRE_HEADER.size:
            magic, kind, generation, size = WIRE_HEADER.unpack_from(self.buffer)
            if magic != WIRE_MAGIC or kind not in (WIRE_CONTROL, WIRE_DATA) or size > MAX_WIRE_FRAME:
                raise ValueError('invalid wire envelope header')
            if kind == WIRE_CONTROL and size != 4096:
                raise ValueError('control envelope must have public fixed size')
            if len(self.buffer) < WIRE_HEADER.size + size:
                break
            out.append((kind, generation, bytes(self.buffer[WIRE_HEADER.size:WIRE_HEADER.size + size])))
            del self.buffer[:WIRE_HEADER.size + size]
        return out


def encode_record(value: dict[str, Any], padded: int = 0) -> bytes:
    body = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    data = b'\x01' + struct.pack('!I', len(body)) + body
    if len(data) > MAX_RECORD:
        raise ValueError('record exceeds transport limit')
    return data + b'\x00' * max(0, padded - len(data))


class RecordDecoder:
    """Incremental framing; zero padding is legal only BETWEEN complete records."""

    def __init__(self) -> None:
        self.buffer = bytearray()

    def feed(self, data: bytes) -> list[dict[str, Any]]:
        self.buffer.extend(data)
        out = []
        while self.buffer:
            if self.buffer[0] == 0:
                index = next((i for i, b in enumerate(self.buffer) if b), len(self.buffer))
                del self.buffer[:index]
                continue
            if self.buffer[0] != 1:
                raise ValueError('invalid record marker')
            if len(self.buffer) < 5:
                break
            n = struct.unpack('!I', self.buffer[1:5])[0]
            if n > MAX_RECORD:
                raise ValueError('oversized record')
            if len(self.buffer) < n + 5:
                break
            out.append(json.loads(self.buffer[5:n + 5]))
            del self.buffer[:n + 5]
        return out


class _Channel:
    def __init__(self, sock: ssl.SSLSocket, handler: Callable[[dict[str, Any]], None]) -> None:
        self.sock, self.handler = sock, handler
        self.tls_version = sock.version()
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.lock, self.send_lock = threading.RLock(), threading.Lock()
        self.io_idle = threading.Condition(self.lock)
        self.plain_sends = 0
        self.queue: deque[bytes] = deque()
        self.pending = self.volume = 0
        self.wire_counts = {'sent_control': 0, 'sent_data': 0, 'received_control': 0, 'received_data': 0}
        self.active = False
        self.generation = -1
        self.cfg = ShapingConfig()
        self.logs: list[dict[str, Any]] = []
        self.cap_exceeded = False
        self.error: str | None = None
        self.decoder = RecordDecoder()
        self.decoder_generation = -1
        self.finished = threading.Event()
        self.finished.set()
        self.timing_failure = False
        self.round_states: dict[int, dict[str, Any]] = {}
        self.sock.setblocking(False)
        self.stopped = threading.Event()
        self.writes: queue.Queue[tuple[bytes, threading.Event, list[str]]] = queue.Queue()
        self.writer: threading.Thread | None = None
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        # SSLSocket is owned ONLY by this thread: no concurrent SSL_read/SSL_write.
        # Application workers submit whole records/ticks through the write queue.
        decoder = WireDecoder()
        pending: tuple[bytes, threading.Event, list[str]] | None = None
        offset = 0
        try:
            while not self.stopped.is_set():
                if pending is None:
                    try:
                        pending = self.writes.get_nowait()
                        offset = 0
                    except queue.Empty:
                        pass
                readable, writable, _ = select.select([self.sock], [self.sock] if pending else [], [], .005)
                if pending is not None and writable:
                    try:
                        offset += self.sock.send(pending[0][offset:])
                        if offset == len(pending[0]):
                            label = 'sent_control' if pending[0][4] == WIRE_CONTROL else 'sent_data'
                            self.wire_counts[label] += len(pending[0])
                            pending[1].set()
                            pending = None
                    except (ssl.SSLWantReadError, ssl.SSLWantWriteError):
                        pass
                if readable or self.sock.pending():
                    try:
                        data = self.sock.recv(65536)
                    except (ssl.SSLWantReadError, ssl.SSLWantWriteError):
                        continue
                    if not data:
                        break
                    for frame in decoder.feed(data):
                        self._dispatch_frame(*frame)
        except Exception as e:
            self.error = repr(e)
        finally:
            self.stopped.set()
            if pending is not None:
                pending[2].append(self.error or 'connection closed')
                pending[1].set()
            while not self.writes.empty():
                try:
                    _, event, errors = self.writes.get_nowait()
                    errors.append(self.error or 'connection closed')
                    event.set()
                except queue.Empty:
                    break
            # The peer may disconnect between configurations. Release its FD
            # here rather than retaining it until the entire server shuts down.
            self.sock.close()

    def _dispatch_frame(self, kind: int, generation: int, payload: bytes) -> None:
        label = 'received_control' if kind == WIRE_CONTROL else 'received_data'
        self.wire_counts[label] += WIRE_HEADER.size + len(payload)
        if kind == WIRE_CONTROL:
            values = RecordDecoder().feed(payload)
            if len(values) != 1 or values[0].get('generation') != generation:
                raise ValueError('invalid control envelope contents')
            self.handler(values[0])
            return
        with self.lock:
            if generation != self.generation:
                return  # Late fragments from a completed/truncated generation never enter a new decoder.
            if generation != self.decoder_generation:
                self.decoder = RecordDecoder()
                self.decoder_generation = generation
            values = self.decoder.feed(payload)
        for value in values:
            self.handler(value)

    def _send_wire(self, data: bytes, *, kind: int = WIRE_DATA, generation: int | None = None,
                   completion: threading.Event | None = None) -> None:
        if self.stopped.is_set():
            raise ConnectionError(self.error or 'connection closed')
        done = completion if completion is not None else threading.Event()
        errors: list[str] = []
        frame = wire_frame(kind, self.generation if generation is None else generation, data)
        self.writes.put((frame, done, errors))
        if threading.current_thread() is self.reader:
            return  # e.g. public ACK; this same I/O loop must perform the write.
        if not done.wait(timeout=5):
            self.error = 'TLS I/O write deadline exceeded'
            self.stopped.set()
            raise TimeoutError(self.error)
        if errors:
            raise ConnectionError(errors[0])

    def send_control(self, value: dict[str, Any], completion: threading.Event | None = None) -> None:
        frame = encode_record(value)
        if len(frame) > 4096:
            raise ValueError('control frame exceeds public fixed size')
        self._send_wire(frame + bytes(4096 - len(frame)), kind=WIRE_CONTROL,
                        generation=value['generation'], completion=completion)

    def start(self, cfg: ShapingConfig, epoch: float, generation: int) -> None:
        if self.stopped.is_set() or self.error:
            raise ConnectionError(self.error or 'connection closed')
        if self.writer and self.writer.is_alive() and not self.finished.is_set():
            raise RuntimeError('sessions must be serialized')
        with self.lock:
            self.queue.clear()
            self.pending = self.volume = 0
            self.logs = []
            self.round_states = {}
            self.cap_exceeded = False
            self.timing_failure = False
            self._last_volume = 0
            self.cfg, self.generation, self.active = cfg, generation, True
            self.finished = threading.Event()
        if cfg.dp:
            self.writer = threading.Thread(target=self._ticks, args=(epoch, generation, self.finished), daemon=True)
            self.writer.start()
        else:
            self.finished.set()

    def enqueue(self, value: dict[str, Any], generation: int, padded: int = 0) -> bool:
        if padded and len(encode_record(value)) > padded:
            raise ValueError('round record exceeds its declared frame size')
        data = encode_record(value, padded)
        if self.cfg.padding:
            p = self.cfg.padding
            data = encode_record(value, int(pad_size(len(data), p.ratio, p.base)))
        if self.cfg.fixed_unit:
            data = encode_record(value, math.ceil(len(data) / self.cfg.fixed_unit) * self.cfg.fixed_unit)
        with self.lock:
            if generation != self.generation or not self.active:
                return False
            if self.cfg.dp:
                assert self.cfg.dp.volume_cap_bytes is not None
                if self.volume + len(data) > self.cfg.dp.volume_cap_bytes:
                    self.cap_exceeded = True
                    return False
                self.volume += len(data)
                self.queue.append(data)
                self.pending += len(data)
                return True
            self.plain_sends += 1
        try:
            self._send_wire(data, generation=generation)
            self.logs.append({'sent_epoch': time.time(), 'application_bytes': len(data), 'kind': value['type']})
        finally:
            with self.io_idle:
                self.plain_sends -= 1
                self.io_idle.notify_all()
        return True

    def _ticks(self, epoch: float, generation: int, finished: threading.Event) -> None:
        d = self.cfg.dp
        assert d and d.delta and d.volume_cap_bytes
        rng = np.random.default_rng()  # OS entropy; never publish a replayable live-noise seed.
        scale = laplace_scale(d.delta, d.eps_dir, math.ceil(d.horizon / d.interval - 1e-12), d.mode)
        mono_epoch = time.monotonic() + epoch - time.time()
        logs = self.logs
        expected_ticks = math.ceil(d.horizon / d.interval - 1e-12)
        try:
            for k in range(expected_ticks):
                target = epoch + min((k + 1) * d.interval, d.horizon)
                mono_target = mono_epoch + min((k + 1) * d.interval, d.horizon)
                time.sleep(max(0., mono_target - time.monotonic()))
                lag = time.monotonic() - mono_target
                if lag > d.interval:
                    self.timing_failure = True
                if time.monotonic() > mono_epoch + d.horizon + d.interval:
                    self.timing_failure = True
                    break  # Do not burst old releases to flush data after the public release window.
                with self.lock:
                    arrival = self.volume - getattr(self, '_last_volume', 0) if k else self.volume
                    self._last_volume = self.volume
                    base = self.pending if d.mode == 'queue' else arrival
                    centre = base + d.bias if d.mode == 'arrivals' else base
                    b = max(0, int(round(centre + rng.laplace(0., scale))))
                    pieces, remaining = [], b
                    while self.queue and remaining:
                        head = self.queue.popleft()
                        take = min(remaining, len(head))
                        pieces.append(head[:take])
                        self.pending -= take
                        remaining -= take
                        if take < len(head):
                            self.queue.appendleft(head[take:])
                    payload = b''.join(pieces) + bytes(remaining)
                sent_at = time.time()
                if b:
                    self._send_wire(payload, generation=generation)
                completion_lag = time.monotonic() - mono_target
                if completion_lag > d.interval:
                    self.timing_failure = True
                logs.append({'tick': k, 'scheduled_epoch': target, 'sent_epoch': sent_at,
                             'lag_s': lag, 'completion_lag_s': completion_lag, 'completed_epoch': time.time(),
                             'application_bytes': b, 'queued_bytes': self.pending})
        except Exception as e:
            self.error = str(e)
        finally:
            with self.lock:
                self.active = False  # discard overflow; NEVER flush after the horizon.
                status = {'type': 'session_end', 'generation': generation, 'expected_ticks': expected_ticks,
                          'completed_ticks': len(logs), 'overflow_bytes': self.pending,
                          'cap_exceeded': self.cap_exceeded, 'error': self.error,
                          'timing_valid': not self.timing_failure and len(logs) == expected_ticks}
            try:
                # The I/O owner sets this event when the public fixed-size end frame is written.
                # An old thread's tail cannot mark a newer generation complete.
                self.send_control(status, completion=finished)
            except Exception as e:
                self.error = str(e)
                finished.set()

    def close(self) -> None:
        self.stopped.set()
        self.reader.join(timeout=6)
        if self.writer:
            self.writer.join(timeout=6)
        self.sock.close()


class TLSToolServer:
    """One persistent TLS link, one tool execution pool, isolated from global services."""

    def __init__(self, host: str, port: int, certfile: str, keyfile: str,
                 tool: Callable[[str, dict[str, Any]], tuple[dict[str, Any], int]]) -> None:
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((host, port))
        self.listener.listen()
        self.address = self.listener.getsockname()
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.minimum_version = ssl.TLSVersion.TLSv1_3
        self.context.load_cert_chain(certfile, keyfile)
        self.tool = tool
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.channels: list[_Channel] = []
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()

    def _accept(self) -> None:
        try:
            while True:
                sock, _ = self.listener.accept()
                # A TCP health probe or rejected TLS peer must not terminate
                # the listener for every subsequent client.
                sock.settimeout(5.)
                try:
                    tls = self.context.wrap_socket(sock, server_side=True)
                except OSError:
                    sock.close()
                    continue
                tls.settimeout(None)
                ready = threading.Event()
                holder: list[_Channel] = []

                def handle(v: dict[str, Any], holder: list[_Channel] = holder, ready: threading.Event = ready) -> None:
                    ready.wait()
                    ch = holder[0]
                    if v['type'] == 'start':
                        ch.start(ShapingConfig.from_dict(v['config']), v['epoch'], v['generation'])
                        ch.send_control({'type': 'ack', 'generation': v['generation']})
                    elif v['type'] == 'call':
                        self.pool.submit(self._execute, ch, v)
                    elif v['type'] == 'round_call':
                        self._start_round(ch, v)
                    elif v['type'] == 'round_poll':
                        self._schedule_round_reply(ch, v)
                    elif v['type'] == 'session_stop':
                        self._end_plain(ch, v)

                ch = _Channel(tls, handle)
                holder.append(ch)
                self.channels.append(ch)
                ready.set()
        except OSError:
            pass

    def _execute(self, ch: _Channel, v: dict[str, Any]) -> None:
        try:
            result, rows = self.tool(v['family'], v['params'])
            response = {'type': 'reply', 'id': v['id'], 'generation': v['generation'], 'result': result, 'matching_rows': rows}
        except Exception as e:
            response = {'type': 'reply', 'id': v['id'], 'generation': v['generation'], 'error': str(e)}
        ch.enqueue(response, v['generation'])

    def _end_plain(self, ch: _Channel, value: dict[str, Any]) -> None:
        if ch.cfg.dp:
            raise ValueError('DP sessions end on their fixed release schedule')

        def complete() -> None:
            with ch.io_idle:
                if value['generation'] != ch.generation:
                    return
                ch.active = False
                if not ch.io_idle.wait_for(lambda: ch.plain_sends == 0, timeout=5):
                    ch.error = 'plain send completion barrier timed out'
                status = {'type': 'session_end', 'generation': value['generation'],
                          'expected_ticks': 0, 'completed_ticks': 0, 'overflow_bytes': 0,
                          'cap_exceeded': ch.cap_exceeded, 'error': ch.error, 'timing_valid': True}
            try:
                ch.send_control(status)
            except Exception as exc:
                ch.error = str(exc)

        # Never block the I/O owner while application workers wait for TLS writes.
        threading.Thread(target=complete, daemon=True).start()

    def _start_round(self, ch: _Channel, v: dict[str, Any]) -> None:
        with ch.lock:
            ch.round_states[v['id']] = {'ready': v['call'] is None, 'payload': b'', 'offset': 0}
        if v['call'] is not None:
            self.pool.submit(self._compute_round, ch, v)
        self._schedule_round_reply(ch, v)

    def _compute_round(self, ch: _Channel, v: dict[str, Any]) -> None:
        try:
            result, rows = self.tool(v['family'], v['call']['params'])
            response = {'result': result, 'matching_rows': rows}
        except Exception as e:
            response = {'error': str(e)}
        payload = json.dumps(response, separators=(',', ':'), allow_nan=False).encode()
        with ch.lock:
            if v['generation'] == ch.generation and v['id'] in ch.round_states:
                ch.round_states[v['id']].update(ready=True, payload=payload)

    def _schedule_round_reply(self, ch: _Channel, v: dict[str, Any]) -> None:
        r = ch.cfg.rounds
        assert r is not None
        # In a DP composition, preserve processing/round delay after actual delivery.
        target = max(v['reply_epoch'], time.time() + r.window) if ch.cfg.dp else v['reply_epoch']

        def send() -> None:
            time.sleep(max(0., target - time.time()))
            with ch.lock:
                if v['generation'] != ch.generation or not ch.active:
                    return
                state = ch.round_states.get(v['id'])
                if state is None:
                    return
                cap = (r.reply_frame - 256) // 2  # hex payload plus a bounded JSON envelope
                chunk = state['payload'][state['offset']:state['offset'] + cap] if state['ready'] else b''
                state['offset'] += len(chunk)
                done = state['ready'] and state['offset'] == len(state['payload'])
            ch.enqueue({'type': 'round_chunk', 'id': v['id'], 'generation': v['generation'],
                        'iteration': v['iteration'], 'payload_hex': chunk.hex(), 'done': done},
                       v['generation'], padded=r.reply_frame)

        threading.Thread(target=send, daemon=True).start()

    def stop(self) -> None:
        self.listener.close()
        for ch in self.channels:
            ch.close()
        self.pool.shutdown(wait=True)


class ShapedTransport:
    """Online calls over long-lived TLS, including experimental synchronized rounds.

    Timing logs and truncated work are diagnostics;
    no empirical privacy or energy claim follows merely from successfully sending bytes.
    """

    def __init__(self, sites: list[tuple[str, int]], cafile: str, cfg: ShapingConfig | None = None,
                 start_lead_s: float = .5) -> None:
        self.cfg = cfg or ShapingConfig()
        if not math.isfinite(start_lead_s) or start_lead_s < .05:
            raise ValueError('start_lead_s must be finite and at least 50 ms')
        self.start_lead_s = start_lead_s
        if self.cfg.rounds:
            r = self.cfg.rounds
            if r.window <= 0 or r.reply_frame < 512 or r.poll_size < 256:
                raise ValueError('live rounds require W > 0, R >= 512 and poll_size >= 256')
            if r.poll_offset is not None and not 0 <= r.poll_offset < r.window:
                raise ValueError('poll_offset must be in [0, W)')
        if self.cfg.dp:
            d = self.cfg.dp
            if d.sensitivity_mode != 'bounded_pair' or not d.volume_cap_bytes or not d.delta or d.delta < 2 * d.volume_cap_bytes:
                raise ValueError('live DP requires explicit bounded-pair Delta >= 2V and an enforced V')
            if d.horizon / d.interval > 10000:
                raise ValueError('too many live ticks')
            if abs(d.horizon / d.interval - round(d.horizon / d.interval)) > 1e-8:
                raise ValueError('live horizon must be an integer multiple of the tick interval')
        self.condition = threading.Condition()
        self.replies: dict[int, dict[str, Any]] = {}
        self.round_chunks: dict[tuple[int, int, int], dict[str, Any]] = {}
        self.active_sites: set[int] = set()
        self.called_sites: set[int] = set()
        self.acks: set[int] = set()
        self.ends: dict[int, dict[str, Any]] = {}
        self.channels = []
        self.generation = 0
        self.next_id = 0
        self.epoch = self.end_epoch = 0.
        self.last_stats: dict[str, Any] = {}
        context = ssl.create_default_context(cafile=cafile)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        try:
            for i, (host, port) in enumerate(sites):
                sock = context.wrap_socket(socket.create_connection((host, port), timeout=30), server_hostname=host)
                sock.settimeout(None)
                def receive(value: dict[str, Any], site: int = i) -> None:
                    self._receive(site, value)

                self.channels.append(_Channel(sock, receive))
        except BaseException:
            self.close()
            raise

    def _receive(self, site: int, value: dict[str, Any]) -> None:
        with self.condition:
            if value.get('generation') != self.generation:
                return
            if value['type'] == 'ack':
                self.acks.add(site)
            elif value['type'] == 'session_end':
                self.ends[site] = value
            elif value['type'] == 'reply':
                self.replies[value['id']] = value
            elif value['type'] == 'round_chunk':
                self.round_chunks[(value['id'], site, value['iteration'])] = value
            self.condition.notify_all()

    def begin(self, called_sites: list[int] | None = None) -> None:
        self._wire_before = {k: sum(ch.wire_counts[k] for ch in self.channels)
                             for k in self.channels[0].wire_counts}
        self.generation += 1
        self.replies, self.acks = {}, set()
        self.ends = {}
        self.round_chunks = {}
        self.called_sites = set(called_sites or [])
        if any(i < 0 or i >= len(self.channels) for i in self.called_sites):
            raise ValueError('declared site outside deployment')
        if self.cfg.rounds:
            if called_sites is None:
                raise ValueError('live rounds need the declared union of called sites for the whole session')
            self.active_sites = draw_active_set(self.called_sites, len(self.channels), self.cfg.rounds.cover_prob,
                                                np.random.default_rng())
        self.epoch = time.time() + self.start_lead_s
        self.end_epoch = self.epoch + (self.cfg.dp.horizon if self.cfg.dp else 30.)
        for ch in self.channels:
            ch.send_control({'type': 'start', 'generation': self.generation, 'epoch': self.epoch, 'config': self.cfg.to_dict()})
            ch.start(self.cfg, self.epoch, self.generation)
        with self.condition:
            if not self.condition.wait_for(lambda: len(self.acks) == len(self.channels), timeout=.8 * self.start_lead_s):
                raise TimeoutError('all-site start acknowledgement missed public epoch')
        time.sleep(max(0., self.epoch - time.time()))

    def call_round(self, family: str, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.cfg.rounds:
            return self._call_synchronized_round(family, calls)
        ids = []
        for call in calls:
            i, ident = int(call['site']), self.next_id
            self.next_id += 1
            ids.append(ident)
            if not self.channels[i].enqueue({'type': 'call', 'id': ident, 'generation': self.generation,
                                              'family': family, 'params': call['params']}, self.generation):
                raise RuntimeError('input cap exceeded or session ended')
        with self.condition:
            if not self.condition.wait_for(lambda: all(i in self.replies for i in ids),
                                           timeout=max(0., self.end_epoch + .2 - time.time())):
                raise TimeoutError('fixed horizon expired; queued bytes were not flushed')
            result = [self.replies[i] for i in ids]
        for r in result:
            if 'error' in r:
                raise RuntimeError(r['error'])
        return result

    def _call_synchronized_round(self, family: str, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        r = self.cfg.rounds
        assert r is not None
        by_site = {int(c['site']): c for c in calls}
        if len(by_site) != len(calls) or not set(by_site) <= self.called_sites:
            raise ValueError('round must contain at most one call per declared site')
        if not calls:
            return []
        rid = self.next_id
        self.next_id += 1
        b = self.epoch + math.ceil((time.time() - self.epoch) / r.window) * r.window
        envelopes = {i: {'type': 'round_call', 'id': rid, 'generation': self.generation, 'family': family,
                          'call': by_site.get(i), 'reply_epoch': b + r.window, 'iteration': 0} for i in self.active_sites}
        common_size = pad_size(max(len(encode_record(v)) for v in envelopes.values()), r.call_ratio, r.call_base)
        time.sleep(max(0., b - time.time()))
        for i, envelope in envelopes.items():
            if not self.channels[i].enqueue(envelope, self.generation, padded=common_size):
                raise RuntimeError('round call cap exceeded or horizon ended')
        payloads: dict[int, bytes] = {i: b'' for i in by_site}
        complete: set[int] = set()
        iteration = 0
        while True:
            with self.condition:
                def arrived(iteration: int = iteration) -> bool:
                    return all((rid, i, iteration) in self.round_chunks for i in self.active_sites)

                if not self.condition.wait_for(arrived, timeout=max(0., self.end_epoch + .2 - time.time())):
                    raise TimeoutError('round replies incomplete at fixed horizon; no flush')
                for i in by_site:
                    frame = self.round_chunks[(rid, i, iteration)]
                    payloads[i] += bytes.fromhex(frame['payload_hex'])
                    if frame['done']:
                        complete.add(i)
            if complete == set(by_site):
                result = [json.loads(payloads[int(c['site'])]) for c in calls]
                for value in result:
                    if 'error' in value:
                        raise RuntimeError(value['error'])
                return result
            iteration += 1
            reply_at = self.epoch + (math.ceil((time.time() - self.epoch) / r.window) + 1) * r.window
            offset = r.window / 2 if r.poll_offset is None else r.poll_offset
            time.sleep(max(0., reply_at - offset - time.time()))
            for i in self.active_sites:
                if not self.channels[i].enqueue({'type': 'round_poll', 'id': rid, 'generation': self.generation,
                                                 'iteration': iteration, 'reply_epoch': reply_at},
                                                self.generation, padded=r.poll_size):
                    raise RuntimeError('round poll cap exceeded or horizon ended')

    def finish(self) -> dict[str, Any]:
        completion_ok = True
        if self.cfg.dp:
            # Completion is a protocol barrier, not an assumed scheduling delay.
            # The bounded wait collects late/error diagnostics; it does not extend the release schedule.
            deadline = time.monotonic() + max(0., self.end_epoch - time.time()) + 6.
            for ch in self.channels:
                if not ch.finished.wait(timeout=max(0., deadline - time.monotonic())):
                    completion_ok = False
            with self.condition:
                completion_ok &= self.condition.wait_for(lambda: len(self.ends) == len(self.channels),
                                                          timeout=max(0., deadline - time.monotonic()))
        else:
            self.end_epoch = time.time()
            for ch in self.channels:
                ch.send_control({'type': 'session_stop', 'generation': self.generation})
            with self.condition:
                completion_ok = self.condition.wait_for(lambda: len(self.ends) == len(self.channels), timeout=6.)
        completion_epoch = time.time()
        self.last_stats = {'epoch': self.epoch, 'end_epoch': completion_epoch,
                           'scheduled_end_epoch': self.end_epoch, 'completion_epoch': completion_epoch,
                           'config_id': self.cfg.config_id(),
                           'upstream': [{'volume_bytes': c.volume, 'overflow_bytes': c.pending,
                                         'cap_exceeded': c.cap_exceeded, 'error': c.error, 'ticks': copy.deepcopy(c.logs),
                                         'timing_valid': not c.timing_failure,
                                         'producer_complete': c.finished.is_set()} for c in self.channels],
                           'downstream': copy.deepcopy(self.ends), 'completion_ok': completion_ok,
                           'wire_protocol': 'GSW1', 'wire_frame_header_bytes': WIRE_HEADER.size,
                           'public_control_frame_bytes': 4096 + WIRE_HEADER.size,
                           'scope': 'tool-execution links only; planning and LLM summary excluded',
                           'privacy_certified': False,
                           'framed_plaintext_bytes': {k: sum(ch.wire_counts[k] for ch in self.channels) - v
                                                      for k, v in self._wire_before.items()},
                           'framed_counter_scope': 'TLS plaintext including GSW1 framing; not IP bytes or energy'}
        self.last_stats['measurement_valid'] = completion_ok and all(
            not c.error and not c.cap_exceeded and not c.timing_failure and not c.pending for c in self.channels)
        self.last_stats['measurement_valid'] &= len(self.ends) == len(self.channels) and all(
            v['timing_valid'] and not v['error'] and not v['cap_exceeded'] and not v['overflow_bytes']
            for v in self.ends.values())
        return self.last_stats

    def close(self) -> None:
        for c in self.channels:
            c.close()
