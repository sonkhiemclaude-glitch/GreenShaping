import socket
import struct

import pytest

from greenshaping.data.wire_capture import audit, packets

PEERS = ('172.31.34.237', '172.31.38.41')


def fixture(path, rows):
    with path.open('wb') as f:
        f.write(struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 128, 1))
        for proto, size, reverse, fragment in rows:
            src, dst = PEERS[::-1] if reverse else PEERS
            ip = struct.pack('!BBHHHBBH4s4s', 0x45, 0, size, 0, fragment, 64, proto, 0,
                             socket.inet_aton(src), socket.inet_aton(dst))
            body = bytearray(struct.pack('!HH', 4567, 5555) + bytes(size - 24))
            if proto == 6:body[12] = 5 << 4
            frame = bytes(12) + b'\x08\x00' + ip + body
            captured = frame[:128]
            f.write(struct.pack('<IIII', 100, 500, len(captured), len(frame)))
            f.write(captured)


def test_common_observer_keeps_tcp_acks_and_quic_udp_without_site_or_payload_labels(tmp_path):
    p = tmp_path / 'trace.pcap'
    fixture(p, [(6, 40, False, 0), (17, 1200, True, 0)])
    rows = list(packets(p, PEERS))
    assert [r['network_bytes'] for r in rows] == [40, 1200]
    assert rows[1]['direction'] == 'down' and rows[1]['observer_link'] == 0
    result = audit(p, PEERS)
    assert result['protocol_counts'] == {'TCP': 1, 'UDP': 1}
    assert result['network_bytes'] == 1240 and result['wire_size_screen_passed']
    assert not result['logical_site_mapping_available']


def test_partial_capture_and_fragmentation_are_not_silently_dropped(tmp_path):
    p = tmp_path / 'trace.pcap'
    fixture(p, [(17, 500, False, 0)])
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError, match='interrupted'):
        list(packets(p, PEERS))
    fixture(p, [(17, 100, False, 0x2000)])
    with pytest.raises(ValueError, match='fragmented'):
        list(packets(p, PEERS))


def test_offloaded_superpackets_are_flagged_instead_of_clipped(tmp_path):
    p = tmp_path / 'trace.pcap'
    fixture(p, [(6, 64000, False, 0)])
    result = audit(p, PEERS, mtu=9001)
    assert result['network_bytes'] == 64000 and result['packets_over_mtu'] == 1
    assert not result['wire_size_screen_passed']


def test_integrity_fingerprint_ignores_ttl_and_header_checksum_but_keeps_ciphertext_prefix(tmp_path):
    p=tmp_path/'trace.pcap';fixture(p,[(17,1200,False,0)])
    original=list(packets(p,PEERS,fingerprints=True))[0]['fingerprint']
    data=bytearray(p.read_bytes());start=24+16+14
    data[start+8]=63;data[start+10:start+12]=b'\x12\x34';p.write_bytes(data)
    assert list(packets(p,PEERS,fingerprints=True))[0]['fingerprint']==original
    data[start+20+8]=1;p.write_bytes(data)
    assert list(packets(p,PEERS,fingerprints=True))[0]['fingerprint']!=original


@pytest.mark.parametrize('link,family', [(0, struct.pack('<I', 2)), (0, struct.pack('>I', 2)), (108, struct.pack('!I', 2))])
def test_bsd_loopback_observations_keep_ip_lengths(tmp_path, link, family):
    p = tmp_path / 'loopback.pcap'
    fixture(p, [(6, 40, False, 0)])
    data = p.read_bytes()
    header = bytearray(data[:24])
    header[20:24] = struct.pack('<I', link)
    old = data[40:]
    frame = family + old[14:]
    p.write_bytes(header + struct.pack('<IIII', 100, 500, len(frame), len(frame)) + frame)
    row, = list(packets(p, PEERS))
    assert row['network_bytes'] == 40 and row['transport'] == 'TCP'
