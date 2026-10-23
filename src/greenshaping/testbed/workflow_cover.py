"""Experimental fixed-envelope TLS adapter. Not a wire-privacy certificate.

Persistent TLS, fixed-size application frames, fixed public schedule. Callback
work is separated from the sending clock. All logs describe acquisition only.
"""
from __future__ import annotations
import hashlib,json,socket,ssl,struct,threading,time
from dataclasses import asdict
from ..controller.workflow_contract import WorkflowEnvelope

START_BYTES=128

def encode_fixed(value,size):
    if size<4:raise ValueError('frame too small')
    data=b'' if value is None else json.dumps(value,separators=(',',':'),allow_nan=False).encode()
    if len(data)>size-4:raise ValueError('payload exceeds public frame cap')
    return struct.pack('!I',len(data))+data+bytes(size-4-len(data))

def decode_fixed(data):
    if len(data)<4:raise ValueError('short frame')
    n=struct.unpack('!I',data[:4])[0]
    if n>len(data)-4:raise ValueError('invalid encrypted length')
    return json.loads(data[4:4+n]) if n else None

def receive(sock,n):
    data=bytearray()
    while len(data)<n:
        part=sock.recv(n-len(data))
        if not part:raise ConnectionError('early end of fixed schedule')
        data.extend(part)
    return bytes(data)

def identity(envelope):return hashlib.sha256(json.dumps(asdict(envelope),sort_keys=True).encode()).digest()

def wait_until(target):time.sleep(max(0.,target-time.monotonic()))

class EnvelopeTLSSite:
    def __init__(self,envelope:WorkflowEnvelope,index,cert,key,handler,host='127.0.0.1'):
        self.envelope=envelope;self.index=index;self.handler=handler;self.logs=[];self.errors=[]
        self.context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);self.context.minimum_version=ssl.TLSVersion.TLSv1_3;self.context.load_cert_chain(str(cert),str(key));self.context.options|=ssl.OP_NO_TICKET
        self.listener=socket.socket();self.listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);self.listener.bind((host,0));self.listener.listen(1);self.listener.settimeout(15);self.address=self.listener.getsockname();self.thread=threading.Thread(target=self._serve,daemon=True);self.thread.start()
    def _serve(self):
        e=self.envelope;ready={};lock=threading.Lock();idx=self.index
        try:
            raw,_=self.listener.accept();raw.settimeout(15)
            with self.context.wrap_socket(raw,server_side=True) as tls:
                start=receive(tls,START_BYTES);epoch=struct.unpack('!d',start[:8])[0]
                if start[8:40]!=identity(e):raise ValueError('public contract mismatch')
                mono=time.monotonic()+epoch-time.time();tls.settimeout(max(5.,e.horizon+3))
                def calculate(r,value):
                    began=time.monotonic()
                    try:
                        answer={'result':self.handler(value)}
                        frame=encode_fixed(answer,e.reply_caps[idx])
                        if time.monotonic()-began>e.compute_bound:self.errors.append('compute_bound')
                    except Exception as exc:
                        self.errors.append(type(exc).__name__);frame=encode_fixed(None,e.reply_caps[idx])
                    with lock:ready[r]=frame
                def reader():
                    try:
                        for r in range(e.rounds):
                            value=decode_fixed(receive(tls,e.call_caps[idx]))
                            if value is None:
                                with lock:ready[r]=encode_fixed(None,e.reply_caps[idx])
                            else:threading.Thread(target=calculate,args=(r,value),daemon=True).start()
                    except Exception as exc:self.errors.append('read:'+type(exc).__name__)
                rx=threading.Thread(target=reader,daemon=True);rx.start()
                for r in range(e.rounds):
                    target=mono+(r+1)*e.round_duration;wait_until(target)
                    with lock:frame=ready.get(r)
                    if frame is None:self.errors.append('reply_not_ready');frame=encode_fixed(None,e.reply_caps[idx])
                    sent=time.time();tls.sendall(frame);self.logs.append({'round':r,'site':idx,'direction':'down','application_bytes':len(frame),'scheduled_epoch':epoch+(r+1)*e.round_duration,'sent_epoch':sent,'lag_s':max(0.,time.monotonic()-target)})
                wait_until(mono+e.horizon+e.interval);rx.join(timeout=.5)
                if rx.is_alive():self.errors.append('reader_not_finished')
        except Exception as exc:self.errors.append(type(exc).__name__)
        finally:self.listener.close()
    def close(self):
        self.thread.join(timeout=max(5.,self.envelope.horizon+3))
        if self.thread.is_alive():self.listener.close();raise TimeoutError('site schedule did not finish')

def run_envelope_tls(envelope,addresses,cafile,planner,*,start_lead=.4):
    """planner(round, previous site->reply) returns request/None per site.

    Planner work may depend on private prior replies, while clocks do not wait
    for it. Violations remain failures and produce cover instead of late flush.
    This local prototype does not force-stop a hung callback; use bounded tools.
    """
    e=envelope;N=len(e.call_caps)
    if len(addresses)!=N:raise ValueError('endpoint count mismatch')
    context=ssl.create_default_context(cafile=str(cafile));context.minimum_version=ssl.TLSVersion.TLSv1_3;socks=[];errors=[];logs=[];responses=[{} for r in range(e.rounds)];events=[threading.Event() for r in range(e.rounds)];plans={};expected={};received_counts=[0]*N;lock=threading.Lock()
    try:
        for host,port in addresses:
            raw=socket.create_connection((host,port),timeout=10);tls=context.wrap_socket(raw,server_hostname=host);tls.settimeout(e.horizon+5);socks.append(tls)
        epoch=time.time()+start_lead;mono=time.monotonic()+epoch-time.time();control=struct.pack('!d',epoch)+identity(e)+bytes(START_BYTES-40)
        for tls in socks:tls.sendall(control)
        def reader(i):
            try:
                for r in range(e.rounds):
                    result=decode_fixed(receive(socks[i],e.reply_caps[i]))
                    received_counts[i]+=e.reply_caps[i]
                    with lock:
                        if expected.get(r,[False]*N)[i] and result is None:errors.append('missing_real_reply')
                        responses[r][i]=result
                        if len(responses[r])==N:events[r].set()
            except Exception as exc:errors.append('read:'+type(exc).__name__)
        def make_plans():
            for r in range(e.rounds):
                if r and not events[r-1].wait(timeout=e.horizon+2):errors.append('prior_reply_timeout');return
                began=time.monotonic()
                try:
                    previous={} if r==0 else dict(responses[r-1]);values=planner(r,previous)
                    if len(values)!=N:raise ValueError('plan endpoint count')
                    frames=[encode_fixed(value,size) for value,size in zip(values,e.call_caps)]
                    if time.monotonic()-began>e.planning_bound:errors.append('planning_bound')
                except Exception as exc:errors.append('plan:'+type(exc).__name__);frames=[encode_fixed(None,n) for n in e.call_caps]
                with lock:
                    plans[r]=frames
                    expected[r]=[decode_fixed(frame) is not None for frame in frames]
        def writer(i):
            try:
                for r in range(e.rounds):
                    offset=r*e.round_duration+e.delivery_bound+e.planning_bound+e.interval;target=mono+offset;wait_until(target)
                    with lock:frames=plans.get(r)
                    if frames is None:errors.append('plan_not_ready');frame=encode_fixed(None,e.call_caps[i])
                    else:frame=frames[i]
                    sent=time.time();socks[i].sendall(frame);logs.append({'round':r,'site':i,'direction':'up','application_bytes':len(frame),'scheduled_epoch':epoch+offset,'sent_epoch':sent,'lag_s':max(0.,time.monotonic()-target)})
            except Exception as exc:errors.append('write:'+type(exc).__name__)
        readers=[threading.Thread(target=reader,args=(i,),daemon=True) for i in range(N)];writers=[threading.Thread(target=writer,args=(i,),daemon=True) for i in range(N)];plan_thread=threading.Thread(target=make_plans,daemon=True)
        for t in readers:t.start()
        plan_thread.start()
        for t in writers:t.start()
        wait_until(mono+e.horizon+e.interval)
        for t in writers+readers+[plan_thread]:
            t.join(timeout=.5)
            if t.is_alive():errors.append('thread_not_finished')
        return {'responses':responses,'client_log':logs,'errors':errors,'start_epoch':epoch,'horizon_s':e.horizon,'control_application_bytes':N*START_BYTES,'completed_frame_application_bytes':sum(r['application_bytes'] for r in logs)+sum(received_counts),'completed':not errors,'client_timing_valid':all(r['lag_s']<=e.interval for r in logs),'scope':'prototype TLS fixed application frames; not packet-level privacy or measured energy'}
    finally:
        for tls in socks:
            try:tls.close()
            except OSError:pass
