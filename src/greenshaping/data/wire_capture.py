"""Protocol-neutral IPv4 observations from classic pcap, including QUIC/UDP.

Payload is never inspected. Sizes are IP packet lengths, including transport
headers and pure ACKs. Offloaded superpackets are flagged, never clipped away.
The encrypted multiplexed observer link is not an individual logical site.
"""
from __future__ import annotations

import hashlib
import socket
import struct
from collections import Counter
from pathlib import Path

MAGICS = {b'\xd4\xc3\xb2\xa1': ('<', 1e6), b'\xa1\xb2\xc3\xd4': ('>', 1e6),
          b'\x4d\x3c\xb2\xa1': ('<', 1e9), b'\xa1\xb2\x3c\x4d': ('>', 1e9)}


def packets(path: Path, peers: tuple[str, str], *, fingerprints: bool = False):
    with Path(path).open('rb') as stream:
        header = stream.read(24)
        if len(header) != 24 or header[:4] not in MAGICS:
            raise ValueError('classic pcap required; invalid or unsupported header')
        endian, resolution = MAGICS[header[:4]]
        _, _, _, _, snaplen, link = struct.unpack(endian + 'HHIIII', header[4:])
        if link not in (0, 1, 101, 108, 113, 228, 276):
            raise ValueError('unsupported link type: ' + str(link))
        while raw := stream.read(16):
            if len(raw) != 16:
                raise ValueError('interrupted pcap record header')
            sec, fraction, captured, original = struct.unpack(endian + 'IIII', raw)
            if captured > snaplen or captured > original or captured > 1024 * 1024 or fraction >= resolution:
                raise ValueError('invalid pcap record lengths or timestamp')
            data = stream.read(captured)
            if len(data) != captured:
                raise ValueError('interrupted pcap record body')
            if link in (0, 108):
                # BSD/macOS loopback: DLT_NULL uses host-order AF_INET;
                # DLT_LOOP uses network byte order. No payload inspection.
                if len(data) < 4:
                    raise ValueError('truncated loopback header')
                families = (struct.pack('!I', 2),) if link == 108 else (struct.pack('<I', 2), struct.pack('>I', 2))
                if data[:4] not in families:
                    continue
                offset = 4
            elif link == 1:
                if len(data) < 14:
                    raise ValueError('truncated Ethernet header')
                offset, ethertype = 14, struct.unpack('!H', data[12:14])[0]
                while ethertype in (0x8100, 0x88A8):
                    if len(data) < offset + 4:
                        raise ValueError('truncated VLAN header')
                    ethertype = struct.unpack('!H', data[offset + 2:offset + 4])[0]
                    offset += 4
                if ethertype != 0x0800:
                    continue
            elif link in (113, 276):
                offset = 16 if link == 113 else 20
                protocol_offset = 14 if link == 113 else 0
                if len(data) < offset:
                    raise ValueError('truncated Linux cooked header')
                if struct.unpack('!H', data[protocol_offset:protocol_offset + 2])[0] != 0x0800:
                    continue
            else:
                offset = 0
            ip = data[offset:]
            if len(ip) < 20 or ip[0] >> 4 != 4:
                raise ValueError('truncated or non-IPv4 packet')
            src, dst = socket.inet_ntoa(ip[12:16]), socket.inet_ntoa(ip[16:20])
            if (src, dst) not in (peers, peers[::-1]):
                continue
            size = struct.unpack('!H', ip[2:4])[0]
            ihl = (ip[0] & 15) * 4
            if ihl < 20 or size < ihl or len(ip) < ihl or original < offset + size:
                raise ValueError('invalid IPv4 packet length')
            if struct.unpack('!H', ip[6:8])[0] & 0x3fff:
                raise ValueError('fragmented observer traffic requires a separate validated adapter')
            protocol = ip[9]
            if protocol not in (6, 17):
                continue
            needed = 20 if protocol == 6 else 8
            if len(ip) < ihl + needed or size < ihl + needed:
                raise ValueError('truncated transport header')
            if protocol == 6:
                tcp_header = (ip[ihl + 12] >> 4) * 4
                if tcp_header < 20 or size < ihl + tcp_header or len(ip) < ihl + tcp_header:
                    raise ValueError('invalid or truncated TCP options/header')
            sport, dport = struct.unpack('!HH', ip[ihl:ihl + 4])
            row = {'epoch': sec + fraction / resolution, 'src': src, 'dst': dst,
                   'source_port': sport, 'destination_port': dport, 'transport': 'TCP' if protocol == 6 else 'UDP',
                   'direction': 'up' if src == peers[0] else 'down', 'network_bytes': size,
                   'captured_frame_bytes': original, 'observer_link': 0}
            if fingerprints:
                # Integrity comparison only: stable metadata and a hash of the
                # captured encrypted UDP prefix. Never an attacker feature.
                identity = ip[2:6] + ip[9:10] + ip[12:20] + ip[ihl:ihl + 4]
                if protocol == 6:
                    identity += ip[ihl + 4:ihl + 14]  # seq, ack, offset/flags; no checksum
                else:
                    identity += ip[ihl + 8:ihl + 40]  # opaque QUIC ciphertext prefix
                row['fingerprint'] = hashlib.sha256(identity).hexdigest()
                row['ip_id'] = struct.unpack('!H', ip[4:6])[0]
                if protocol == 6:
                    row['tcp_seq'], row['tcp_ack'] = struct.unpack('!II', ip[ihl + 4:ihl + 12])
                    row['tcp_flags'] = ip[ihl + 13]
                    row['tcp_payload_bytes'] = size - ihl - (ip[ihl + 12] >> 4) * 4
            yield row


def audit(path: Path, peers: tuple[str, str], mtu: int = 9001) -> dict:
    counts, directions, flows = Counter(), Counter(), Counter()
    total, oversize = 0, 0
    start, end = None, None
    for row in packets(path, peers):
        counts[row['transport']] += 1
        directions[row['direction']] += 1
        flows[f"{row['transport']}:{row['source_port']}->{row['destination_port']}"] += 1
        total += row['network_bytes']
        oversize += row['network_bytes'] > mtu
        start = row['epoch'] if start is None else min(start, row['epoch'])
        end = row['epoch'] if end is None else max(end, row['epoch'])
    return {'packets': sum(counts.values()), 'protocol_counts': dict(counts), 'direction_counts': dict(directions),
            'flows': dict(flows), 'network_bytes': total, 'start_epoch': start, 'end_epoch': end,
            'packets_over_mtu': oversize, 'configured_mtu': mtu, 'wire_size_screen_passed': not oversize and bool(counts),
            'logical_site_mapping_available': False, 'privacy_evaluation_performed': False}
