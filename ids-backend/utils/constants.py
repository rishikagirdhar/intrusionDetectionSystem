# utils/constants.py

# Model thresholds
MODEL1_THRESHOLD = 0.8  # Minimum confidence to trigger attack classification
ATTACK_THRESHOLDS = {
    'DoS': 0.95,    # High threshold for common attacks
    'Probe': 0.90,
    'R2L': 0.85,
    'U2R': 0.80     # Lower threshold for rare attacks
}

# Feature names (from KDD dataset)
NUMERICAL_FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

CATEGORICAL_FEATURES = [
    'protocol_type',  # tcp, udp, icmp
    'service',       # http, smtp, ftp, etc.
    'flag'           # SF, REJ, etc.
]

# Action mappings
ACTIONS = {
    'ALLOW': 0,
    'ALERT_ADMIN': 1,
    'BLOCK': 2
}

# Attack type mappings (must match your model's classes)
ATTACK_TYPES = {
    'normal': 0,
    'DoS': 1,
    'Probe': 2,
    'R2L': 3,
    'U2R': 4
}