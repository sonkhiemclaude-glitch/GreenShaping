import subprocess
import pytest
from greenshaping.testbed.minimal_tls import MinimalServer, MinimalClient
from greenshaping.testbed.transport import ShapedTransport, TLSToolServer
from greenshaping.shaping.pipeline import ShapingConfig


def test_persistent_minimal_baseline_and_control_decomposition(tmp_path):
    cert,key=tmp_path/'cert',tmp_path/'key'
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=127.0.0.1',
                    '-addext','subjectAltName=IP:127.0.0.1','-keyout',str(key),'-out',str(cert)],capture_output=True,check=True)
    def tool(f,p):return {'count':p['n']},p['n']
    server=MinimalServer(str(cert),str(key),tool)
    shaped=TLSToolServer('127.0.0.1',0,str(cert),str(key),tool)
    a=b=None
    try:
        a=MinimalClient([server.address]*2,str(cert))
        for n in [3,7]:
            assert a.call_round('record_count',[{'site':0,'params':{'n':n}}])[0]['result']['count']==n
        assert a.tx>0 and a.rx>0
        b=ShapedTransport([shaped.address]*2,str(cert),ShapingConfig(),start_lead_s=1.)
        for _ in range(2):
            b.begin([0]);b.call_round('record_count',[{'site':0,'params':{'n':3}}]);stats=b.finish()
            assert stats['measurement_valid']
            c=stats['framed_plaintext_bytes']
            assert c['sent_control']==2*2*4113 and c['received_control']==2*2*4113
            assert c['sent_data']>0 and c['received_data']>0
    finally:
        if a:a.close()
        if b:b.close()
        server.close();shaped.stop()


def test_shaped_server_releases_disconnected_peer_descriptors(tmp_path):
    import time
    cert,key=tmp_path/'cert',tmp_path/'key'
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=127.0.0.1',
                    '-addext','subjectAltName=IP:127.0.0.1','-keyout',str(key),'-out',str(cert)],capture_output=True,check=True)
    server=TLSToolServer('127.0.0.1',0,str(cert),str(key),lambda f,p: ({},0))
    try:
        for _ in range(12):
            client=ShapedTransport([server.address]*3,str(cert))
            client.close()
        deadline=time.monotonic()+3
        while time.monotonic()<deadline and any(c.sock.fileno()!=-1 for c in server.channels):
            time.sleep(.01)
        assert len(server.channels)==36
        assert all(c.sock.fileno()==-1 for c in server.channels)
        assert server.thread.is_alive()
    finally:
        server.stop()
