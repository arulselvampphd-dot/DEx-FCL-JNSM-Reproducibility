from __future__ import annotations
import re

CICIOT_GROUPS = ["Benign", "DDoS", "DoS", "Recon", "Spoofing", "BruteForce", "WebBased", "Mirai"]
EDGE_GROUPS = ["Normal", "DDoS", "Scanning", "MITM", "Injection", "Malware"]

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def map_ciciot_label(label: str) -> str:
    x = _norm(label)
    if "benign" in x: return "Benign"
    if x.startswith("ddos"): return "DDoS"
    if x.startswith("dos"): return "DoS"
    if x.startswith("recon") or "vulnerabilityscan" in x: return "Recon"
    if "spoof" in x or "mitmarp" in x: return "Spoofing"
    if "bruteforce" in x: return "BruteForce"
    if x.startswith("mirai"): return "Mirai"
    web_tokens = ["browserhijacking","commandinjection","sqlinjection","xss","backdoormalware","uploadingattack"]
    if any(t in x for t in web_tokens): return "WebBased"
    raise ValueError(f"Unmapped CICIoT2023 label: {label!r}")

def map_edge_label(label: str) -> str:
    x = _norm(label)
    if x in {"normal","benign","0"}: return "Normal"
    if "ddos" in x or x.startswith("dos"): return "DDoS"
    if any(t in x for t in ["fingerprint","portscann","vulnerabilityscanner","vulnerabilityscan","scanning"]): return "Scanning"
    if "mitm" in x or "spoof" in x: return "MITM"
    if any(t in x for t in ["sqlinjection","xss","uploading"]): return "Injection"
    if any(t in x for t in ["backdoor","ransomware","password"]): return "Malware"
    raise ValueError(f"Unmapped Edge-IIoTset label: {label!r}")

CICIOT_CONTINUAL = [
    ["Benign", "DDoS", "DoS"], ["Recon"], ["Spoofing"], ["BruteForce"], ["WebBased"], ["Mirai"]
]
EDGE_CONTINUAL = [
    ["Normal", "DDoS"], ["Scanning"], ["Injection"], ["Malware"], ["MITM"]
]
