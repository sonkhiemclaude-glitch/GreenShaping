"""Command builders for capture and export. They RETURN argument lists; nothing is executed here.

Run them yourself on the testbed host (tcpdump needs capture privileges). No shell
strings are built from config values, so configs cannot inject commands.
"""
from __future__ import annotations


def tcpdump_command(interface: str, out_pcap: str, ports: list[int]) -> list[str]:
    if not ports:
        raise ValueError("need at least one site port")
    flt = " or ".join(f"tcp port {int(p)}" for p in ports)
    return ["tcpdump", "-i", interface, "-w", out_pcap, "-s", "128", flt]


def tshark_export_command(in_pcap: str) -> list[str]:
    """Export one row per TCP segment with payload, in the column order CaptureCsvAdapter expects."""
    return ["tshark", "-r", in_pcap, "-Y", "tcp.len>0", "-T", "fields", "-E", "separator=,",
            "-e", "frame.time_epoch", "-e", "ip.src", "-e", "ip.dst",
            "-e", "tcp.srcport", "-e", "tcp.dstport", "-e", "tcp.len"]
