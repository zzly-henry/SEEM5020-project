"""
Real-world dataset loader for Strict Turnstile + α-Bounded Deletion streams.

Supported dataset:
  MAWI Working Group Traffic Archive — destination IP as element.
  File: 202506181400.pcap

Each packet is treated as a +1 insertion on the destination IP. Deletions are
then sampled from previously seen items to enforce α-bounded deletion and
strict turnstile (f_e >= 0 always).

To use:
  Place 202506181400.pcap in the data/ folder, or pass the full path.

Download (if needed):
  https://mawi.wide.ad.jp/mawi/samplepoint-F/2025/202506181400.html
"""

import os
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ======================================================================
# MAWI pcap loader
# ======================================================================

def _read_dst_ips_from_pcap(pcap_path: str, max_packets: int = 2_000_000) -> List[str]:
    """
    Read destination IPs from a pcap file using dpkt (preferred) or scapy.

    Returns a list of destination IP strings, one per packet (up to max_packets).
    """
    dst_ips: List[str] = []

    # Try dpkt first (faster for large pcaps)
    try:
        import dpkt
        import socket

        with open(pcap_path, "rb") as f:
            try:
                pcap_reader = dpkt.pcap.Reader(f)
            except ValueError:
                # Might be pcapng format
                f.seek(0)
                pcap_reader = dpkt.pcapng.Reader(f)

            for ts, buf in pcap_reader:
                if len(dst_ips) >= max_packets:
                    break
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if isinstance(eth.data, dpkt.ip.IP):
                        ip = eth.data
                        dst = socket.inet_ntoa(ip.dst)
                        dst_ips.append(dst)
                    elif isinstance(eth.data, dpkt.ip6.IP6):
                        ip6 = eth.data
                        dst = socket.inet_ntop(socket.AF_INET6, ip6.dst)
                        dst_ips.append(dst)
                except Exception:
                    continue

        if dst_ips:
            return dst_ips

    except ImportError:
        pass

    # Fallback: scapy (slower but more robust)
    try:
        from scapy.all import PcapReader, IP, IPv6
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                if len(dst_ips) >= max_packets:
                    break
                if pkt.haslayer(IP):
                    dst_ips.append(pkt[IP].dst)
                elif pkt.haslayer(IPv6):
                    dst_ips.append(pkt[IPv6].dst)

        if dst_ips:
            return dst_ips

    except ImportError:
        pass

    if not dst_ips:
        raise RuntimeError(
            f"Could not read pcap file '{pcap_path}'. "
            "Install dpkt (`pip install dpkt`) or scapy (`pip install scapy`)."
        )

    return dst_ips


def load_mawi_stream(
    pcap_path: str = None,
    N: int = 500_000,
    alpha: float = 2.0,
    max_packets: int = 2_000_000,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    """
    Load MAWI pcap trace and convert to a strict-turnstile α-bounded deletion stream.

    Each packet → (dst_ip, +1). Deletions are sampled from accumulated items
    to satisfy D <= (1 - 1/α) * I and f_e >= 0 at all times.

    Parameters
    ----------
    pcap_path : str or None
        Path to 202506181400.pcap. If None, looks in data/ folder.
    N : int
        Target total number of stream operations (insertions + deletions).
    alpha : float
        α-bounded deletion parameter (>= 1).
    max_packets : int
        Maximum packets to read from the pcap file.
    seed : int
        Random seed for deletion sampling.

    Returns
    -------
    stream : list of (dst_ip, delta) tuples
    true_freq : dict {dst_ip: final_frequency}
    """
    if pcap_path is None:
        pcap_path = os.path.join(DATA_DIR, "202506181400.pcap")

    if not os.path.isfile(pcap_path):
        raise FileNotFoundError(
            f"MAWI pcap file not found at '{pcap_path}'. "
            f"Please place 202506181400.pcap in the data/ folder or pass the full path."
        )

    print(f"[INFO] Reading MAWI pcap: {pcap_path} (max {max_packets} packets)...")
    raw_ips = _read_dst_ips_from_pcap(pcap_path, max_packets=max_packets)
    print(f"[INFO] Read {len(raw_ips)} destination IPs from pcap.")

    # Build α-bounded turnstile stream
    return _build_turnstile_stream(raw_ips, N, alpha, np.random.RandomState(seed))


# ======================================================================
# Turnstile stream builder (shared logic)
# ======================================================================

def _build_turnstile_stream(
    raw_items: list,
    N: int,
    alpha: float,
    rng: np.random.RandomState,
) -> Tuple[List[Tuple], Dict]:
    """
    Convert a list of raw item occurrences into a strict-turnstile
    α-bounded deletion stream of ~N operations.

    Strategy: treat raw items as insertions, then sample deletions from
    accumulated frequencies to satisfy the α-property.
    """
    if alpha <= 1.0:
        deletion_ratio = 0.0
    else:
        deletion_ratio = 1.0 - 1.0 / alpha

    num_inserts = min(int(N / (1.0 + deletion_ratio)), len(raw_items))
    num_deletes = min(N - num_inserts, int(deletion_ratio * num_inserts))

    insert_items = raw_items[:num_inserts]

    stream: List[Tuple] = []
    current_freq: Counter = Counter()

    insert_idx = 0
    deletes_remaining = num_deletes
    batch_size = max(1, num_inserts // max(num_deletes, 1) + 1)

    while insert_idx < num_inserts or deletes_remaining > 0:
        batch_end = min(insert_idx + batch_size, num_inserts)
        for i in range(insert_idx, batch_end):
            item = insert_items[i]
            stream.append((item, 1))
            current_freq[item] += 1
        insert_idx = batch_end

        if deletes_remaining > 0:
            pos_items = [it for it, c in current_freq.items() if c > 0]
            if pos_items:
                pos_counts = np.array([current_freq[it] for it in pos_items], dtype=np.float64)
                pos_probs = pos_counts / pos_counts.sum()
                n_del = min(deletes_remaining, len(pos_items), batch_size)
                del_indices = rng.choice(len(pos_items), size=n_del, p=pos_probs, replace=True)
                for idx in del_indices:
                    item = pos_items[idx]
                    if current_freq[item] > 0 and deletes_remaining > 0:
                        stream.append((item, -1))
                        current_freq[item] -= 1
                        deletes_remaining -= 1

        if insert_idx >= num_inserts and deletes_remaining > 0:
            while deletes_remaining > 0:
                pos_items = [it for it, c in current_freq.items() if c > 0]
                if not pos_items:
                    break
                pos_counts = np.array([current_freq[it] for it in pos_items], dtype=np.float64)
                pos_probs = pos_counts / pos_counts.sum()
                n_del = min(deletes_remaining, len(pos_items))
                del_indices = rng.choice(len(pos_items), size=n_del, p=pos_probs, replace=True)
                for idx in del_indices:
                    item = pos_items[idx]
                    if current_freq[item] > 0 and deletes_remaining > 0:
                        stream.append((item, -1))
                        current_freq[item] -= 1
                        deletes_remaining -= 1
                if deletes_remaining <= 0:
                    break

    true_freq = {item: cnt for item, cnt in current_freq.items() if cnt > 0}
    return stream, true_freq


def get_stream_stats(stream: list, true_freq: dict) -> dict:
    """Return summary statistics."""
    inserts = sum(1 for _, d in stream if d > 0)
    deletes = sum(1 for _, d in stream if d < 0)
    F1 = sum(true_freq.values())
    distinct = len(true_freq)
    return {
        "total_ops": len(stream),
        "inserts": inserts,
        "deletes": deletes,
        "F1": F1,
        "distinct_elements": distinct,
        "alpha_actual": (inserts + deletes) / F1 if F1 > 0 else float("inf"),
    }