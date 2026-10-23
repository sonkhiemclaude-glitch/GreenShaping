"""Matched local TLS acquisition for envelope and periodic full-frame cover.

Experimental scripted workflows, not a distributed implementation or a wire-DP
certificate. Both policies retain all public slots, including after failure.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import select
import socket
import ssl
import struct
import threading
import time
from dataclasses import asdict

from .workflow_cover import START_BYTES, decode_fixed, encode_fixed, receive, wait_until


class SerializedTLS:
    """Serialize OpenSSL calls without holding a lock across blocking reads.

    Readiness waits use the descriptor outside the lock, so the sending clock
    can continue when the receive side is awaiting the next cover frame.
    """
    def __init__(self, tls, timeout):
        self.tls, self.timeout, self.lock = tls, timeout, threading.Lock()
        self.tls.setblocking(False)

    def _attempt(self, operation, deadline):
        while True:
            want_write = False
            try:
                with self.lock:
                    return operation()
            except ssl.SSLWantReadError:
                pass
            except ssl.SSLWantWriteError:
                want_write = True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('paced TLS I/O deadline')
            select.select([] if want_write else [self.tls], [self.tls] if want_write else [], [], min(.01, remaining))

    def recv(self, n):
        return self._attempt(lambda: self.tls.recv(n), time.monotonic() + self.timeout)

    def sendall(self, data):
        deadline, offset = time.monotonic() + self.timeout, 0
        while offset < len(data):
            count = self._attempt(lambda: self.tls.send(data[offset:]), deadline)
            if count == 0:
                raise ConnectionError('closed TLS send stream')
            offset += count

    def close(self):
        with self.lock:
            self.tls.close()


def slots(envelope, policy, direction):
    if policy == 'envelope':
        return [(r * envelope.round_duration + envelope.planning_bound + envelope.interval
                 if direction == 'up' else (r + 1) * envelope.round_duration)
                for r in range(envelope.rounds)]
    if policy == 'periodic':
        return [(k + 1) * envelope.interval
                for k in range(math.ceil(envelope.horizon / envelope.interval - 1e-10))]
    if policy.startswith('periodic_'):
        count = int(policy.removeprefix('periodic_'))
        if not 1 <= count <= 1000:
            raise ValueError('invalid public periodic tick count')
        return [(k + 1) * envelope.horizon / count for k in range(count)]
    raise ValueError('unknown cover policy')


def contract_hash(envelope, policy):
    if envelope.delivery_bound != 0:
        raise ValueError('this loopback acquisition requires delivery_bound=0')
    return hashlib.sha256(json.dumps({'envelope': asdict(envelope), 'policy': policy},
                                     sort_keys=True).encode()).digest()


class PacedTLSSite:
    def __init__(self, envelope, policy, index, cert, key, handler, port=0):
        self.envelope, self.policy, self.index = envelope, policy, index
        self.handler, self.logs, self.errors, self.workers = handler, [], [], []
        self.identity = contract_hash(envelope, policy)
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.minimum_version = ssl.TLSVersion.TLSv1_3
        self.context.options |= ssl.OP_NO_TICKET
        self.context.load_cert_chain(str(cert), str(key))
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(('127.0.0.1', port))
        self.listener.listen(1)
        self.listener.settimeout(10)
        self.address = self.listener.getsockname()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        e, i = self.envelope, self.index
        ready, pending = {}, collections.deque()
        lock = threading.Lock()
        try:
            raw, _ = self.listener.accept()
            raw.settimeout(10)
            with self.context.wrap_socket(raw, server_side=True) as raw_tls:
                control = receive(raw_tls, START_BYTES)
                epoch = struct.unpack('!d', control[:8])[0]
                if control[8:40] != self.identity:
                    raise ValueError('public contract mismatch')
                mono = time.monotonic() + epoch - time.time()
                tls = SerializedTLS(raw_tls, e.horizon + 3)

                def calculate(message):
                    began = time.monotonic()
                    try:
                        answer = self.handler(message['value'])
                        frame = encode_fixed({'r': message['r'], 'value': answer}, e.reply_caps[i])
                    except Exception as exc:
                        self.errors.append('tool:' + type(exc).__name__)
                        frame = encode_fixed(None, e.reply_caps[i])
                    duration = time.monotonic() - began
                    if duration > e.compute_bound:
                        self.errors.append('compute_bound')
                    with lock:
                        ready[message['r']] = frame
                        pending.append(message['r'])

                def reader():
                    try:
                        for _ in slots(e, self.policy, 'up'):
                            message = decode_fixed(receive(tls, e.call_caps[i]))
                            if message is not None:
                                if type(message.get('r')) is not int or not 0 <= message['r'] < e.rounds:
                                    raise ValueError('invalid logical round')
                                worker = threading.Thread(target=calculate, args=(message,), daemon=True)
                                self.workers.append(worker)
                                worker.start()
                    except Exception as exc:
                        self.errors.append('read:' + type(exc).__name__)

                rx = threading.Thread(target=reader, daemon=True)
                rx.start()
                for k, offset in enumerate(slots(e, self.policy, 'down')):
                    target = mono + offset
                    wait_until(target)
                    with lock:
                        rid = k if self.policy == 'envelope' else (pending.popleft() if pending else None)
                        frame = ready.pop(rid, None)
                    real = frame is not None and decode_fixed(frame) is not None
                    if frame is None:
                        frame = encode_fixed(None, e.reply_caps[i])
                    began = time.time()
                    tls.sendall(frame)
                    self.logs.append({'slot': k, 'site': i, 'direction': 'down',
                                      'application_bytes': len(frame), 'scheduled_epoch': epoch + offset,
                                      'sent_epoch': began, 'lag_s': max(0., time.monotonic() - target),
                                      'internal_real_frame': real})
                wait_until(mono + max(e.horizon, slots(e, self.policy, 'down')[-1]) + e.interval)
                rx.join(timeout=.5)
                if rx.is_alive():
                    self.errors.append('reader_not_finished')
                for worker in self.workers:
                    worker.join(timeout=.5)
                    if worker.is_alive():
                        self.errors.append('tool_not_finished')
        except Exception as exc:
            self.errors.append('site:' + type(exc).__name__)
        finally:
            self.listener.close()

    def close(self):
        self.thread.join(timeout=self.envelope.horizon + 4)
        if self.thread.is_alive():
            self.listener.close()
            raise TimeoutError('site lifecycle did not finish')


def run_paced_tls(envelope, policy, addresses, cafile, planner, actual_rounds, start_lead=.2):
    """Return measured acquisition data; callers must also validate site logs.

    planner consumes private prior replies. Its actual round count never changes
    either public transmission schedule. Only bounded callbacks are supported.
    """
    e, n = envelope, len(envelope.call_caps)
    if type(actual_rounds) is not int or not 1 <= actual_rounds <= e.rounds:
        raise ValueError('actual round count outside public contract')
    if len(addresses) != n:
        raise ValueError('endpoint count mismatch')
    public_id = contract_hash(e, policy)
    context = ssl.create_default_context(cafile=str(cafile))
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    sockets, errors, logs = [], [], []
    queues = [collections.deque() for _ in range(n)]
    responses = [{} for _ in range(actual_rounds)]
    events = [threading.Event() for _ in range(actual_rounds)]
    expected, received_bytes, lock = {}, [0] * n, threading.Lock()
    progress = {'tool_completed_epoch': None, 'planner_durations_s': []}
    try:
        for host, port in addresses:
            raw = socket.create_connection((host, port), timeout=10)
            tls = context.wrap_socket(raw, server_hostname=host)
            tls.settimeout(e.horizon + 3)
            sockets.append(tls)
        epoch = time.time() + start_lead
        mono = time.monotonic() + epoch - time.time()
        end_offset = max(e.horizon, slots(e, policy, 'down')[-1])
        deadline = mono + end_offset + e.interval
        control = struct.pack('!d', epoch) + public_id + bytes(START_BYTES - 40)
        for tls in sockets:
            tls.sendall(control)
        sockets = [SerializedTLS(tls, e.horizon + 3) for tls in sockets]

        def reader(i):
            try:
                for _ in slots(e, policy, 'down'):
                    message = decode_fixed(receive(sockets[i], e.reply_caps[i]))
                    received_bytes[i] += e.reply_caps[i]
                    if message is None:
                        continue
                    r = message['r']
                    with lock:
                        if type(r) is not int or not 0 <= r < actual_rounds or i not in expected.get(r, set()):
                            errors.append('unexpected_reply')
                            continue
                        if i in responses[r]:
                            errors.append('duplicate_reply')
                            continue
                        responses[r][i] = message['value']
                        if set(responses[r]) == expected[r]:
                            if r == actual_rounds - 1:
                                progress['tool_completed_epoch'] = time.time()
                            events[r].set()
            except Exception as exc:
                errors.append('read:' + type(exc).__name__)

        def make_plans():
            for r in range(actual_rounds):
                if r and not events[r - 1].wait(timeout=max(0., deadline - time.monotonic())):
                    errors.append('prior_reply_timeout')
                    return
                began = time.monotonic()
                try:
                    values = planner(r, {} if r == 0 else dict(responses[r - 1]))
                    if len(values) != n:
                        raise ValueError('plan endpoint count')
                    frames = [None if v is None else encode_fixed({'r': r, 'value': v}, cap)
                              for v, cap in zip(values, e.call_caps)]
                    if not any(frame is not None for frame in frames):
                        raise ValueError('at least one real call required')
                except Exception as exc:
                    errors.append('plan:' + type(exc).__name__)
                    return
                duration = time.monotonic() - began
                progress['planner_durations_s'].append(duration)
                if duration > e.planning_bound:
                    errors.append('planning_bound')
                with lock:
                    expected[r] = {i for i, f in enumerate(frames) if f is not None}
                    for i, frame in enumerate(frames):
                        if frame is not None:
                            queues[i].append((r, frame))

        def writer(i):
            try:
                for k, offset in enumerate(slots(e, policy, 'up')):
                    target = mono + offset
                    wait_until(target)
                    with lock:
                        item = None
                        if queues[i] and (policy != 'envelope' or queues[i][0][0] == k):
                            item = queues[i].popleft()
                    frame = encode_fixed(None, e.call_caps[i]) if item is None else item[1]
                    began = time.time()
                    sockets[i].sendall(frame)
                    logs.append({'slot': k, 'site': i, 'direction': 'up',
                                 'application_bytes': len(frame), 'scheduled_epoch': epoch + offset,
                                 'sent_epoch': began, 'lag_s': max(0., time.monotonic() - target),
                                 'internal_real_frame': item is not None})
            except Exception as exc:
                errors.append('write:' + type(exc).__name__)

        readers = [threading.Thread(target=reader, args=(i,), daemon=True) for i in range(n)]
        writers = [threading.Thread(target=writer, args=(i,), daemon=True) for i in range(n)]
        plan_thread = threading.Thread(target=make_plans, daemon=True)
        for thread in readers + [plan_thread] + writers:
            thread.start()
        wait_until(deadline)
        for thread in writers + readers + [plan_thread]:
            thread.join(timeout=.5)
            if thread.is_alive():
                errors.append('thread_not_finished')
        if not events[-1].is_set():
            errors.append('workflow_incomplete')
        elapsed = progress['tool_completed_epoch']
        return {'policy': policy, 'start_epoch': epoch, 'public_horizon_s': end_offset,
                'tool_completion_s': None if elapsed is None else elapsed - epoch,
                'responses': responses, 'errors': errors, 'client_logs': logs,
                'planner_durations_s': progress['planner_durations_s'],
                'application_bytes': sum(x['application_bytes'] for x in logs) + sum(received_bytes),
                'control_application_bytes': n * START_BYTES,
                'completed': not errors and events[-1].is_set()}
    finally:
        for tls in sockets:
            try:
                tls.close()
            except OSError:
                pass
