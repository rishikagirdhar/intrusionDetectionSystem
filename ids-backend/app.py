import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import re
import joblib
import pandas as pd
import numpy as np
import requests

# Load environment variables
load_dotenv()

# Custom CSS for dark theme
def set_dark_theme():
    st.markdown("""
    <style>
    :root {
        --primary-bg: #0E1117;
        --secondary-bg: #262730;
        --text-color: #FAFAFA;
    }
    
    .stApp {
        background-color: var(--primary-bg);
        color: var(--text-color);
    }
    
    .stChatMessage {
        background-color: var(--secondary-bg) !important;
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
    }
    
    .stTextInput input {
        background-color: var(--secondary-bg) !important;
        color: var(--text-color) !important;
    }
    
    .stButton button {
        background-color: #4A4E69;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
    }
    
    .stButton button:hover {
        background-color: #5C5F77;
    }
    
    .stSidebar {
        background-color: var(--secondary-bg) !important;
    }
    
    .stSpinner > div {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

set_dark_theme()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "🛡️ Hello! I'm your IDS/IPS security assistant. Ask me about network attacks, detection methods, or why a connection was blocked."}
    ]
def get_chat_response(messages):
    try:
        response = requests.post(
            'http://localhost:5000/chat',
            json={'messages': messages},
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 200:
            return response.json().get('response', 'No response from backend')
        return f"⚠️ Backend error (HTTP {response.status_code}): {response.text}"
    except Exception as e:
        return f"⚠️ Failed to connect to backend: {str(e)}"

# Constants (keep your existing constants)
ATTACK_THRESHOLDS = {
    'normal': 0.3, 'neptune': 0.45, 'smurf': 0.45, 'back': 0.55, 'teardrop': 0.55,
    'nmap': 0.6, 'satan': 0.6, 'ipsweep': 0.6, 'portsweep': 0.6, 'warezclient': 0.65,
    'warezmaster': 0.65, 'ftp_write': 0.7, 'guess_passwd': 0.7, 'imap': 0.7,
    'multihop': 0.7, 'phf': 0.7, 'spy': 0.7, 'buffer_overflow': 0.8,
    'loadmodule': 0.8, 'perl': 0.8, 'rootkit': 0.8, 'default': 0.6
}

RISK_LEVELS = {
    'normal': 'LOW', 'neptune': 'HIGH', 'smurf': 'HIGH', 'back': 'HIGH',
    'nmap': 'MEDIUM', 'satan': 'MEDIUM', 'ipsweep': 'MEDIUM', 'portsweep': 'MEDIUM',
    'warezclient': 'HIGH', 'warezmaster': 'HIGH', 'ftp_write': 'CRITICAL',
    'guess_passwd': 'CRITICAL', 'imap': 'CRITICAL', 'multihop': 'CRITICAL',
    'phf': 'CRITICAL', 'spy': 'CRITICAL', 'buffer_overflow': 'CRITICAL',
    'loadmodule': 'CRITICAL', 'perl': 'CRITICAL', 'rootkit': 'CRITICAL'
}



# Attack Detection Knowledge Base
ATTACK_DETECTION_METHODS = {
    'neptune': {
        'signatures': [
            "Unusually high SYN packets to a single destination",
            "No complete TCP handshakes (SYN without ACK)",
            "Source IP spoofing patterns",
            "Same source port for multiple connections"
        ],
        'anomalies': [
            "Extremely short connection durations (<0.1s)",
            "Zero bytes transferred in established connections",
            "High ratio of SYN packets to other packet types",
            "Abnormal distribution of packet sizes"
        ],
        'mitigation': [
            "Implement SYN cookies",
            "Rate limit SYN packets per source IP",
            "Enable TCP RST on invalid connections",
            "Configure firewall to block suspicious IPs"
        ]
    },
    'smurf': {
        'signatures': [
            "ICMP echo requests to broadcast addresses",
            "Same source/destination IP in ICMP packets",
            "Abnormally large ICMP packet size",
            "Unusual ICMP payload patterns"
        ],
        'anomalies': [
            "Extremely high ICMP traffic volume",
            "Multiple hosts responding to single ICMP request",
            "Unusual TTL values in ICMP packets",
            "Abnormal timing between request and response"
        ],
        'mitigation': [
            "Disable IP-directed broadcasts at routers",
            "Filter ICMP echo requests to broadcast addresses",
            "Implement ICMP rate limiting",
            "Monitor ICMP traffic ratios"
        ]
    },
    'nmap': {
        'signatures': [
            "Sequential port scanning patterns",
            "TCP SYN/FIN/XMAS/NULL scans",
            "TTL values indicating nmap default settings",
            "Unusual flag combinations in TCP headers"
        ],
        'anomalies': [
            "Multiple ports accessed from single source with no follow-up",
            "Unusual flag combinations in TCP headers",
            "High ratio of unanswered connection attempts",
            "Abnormal timing between scan packets"
        ],
        'mitigation': [
            "Implement port scan detection systems",
            "Rate limit connection attempts per source IP",
            "Use honeyports to detect scanning activity",
            "Configure IDS with scan detection rules"
        ]
    },
    'buffer_overflow': {
        'signatures': [
            "Exceptionally long strings in input fields",
            "NOP sled patterns in network traffic",
            "Known exploit code signatures",
            "Shellcode patterns in payloads"
        ],
        'anomalies': [
            "Process crashes following specific inputs",
            "Unexpected privilege escalation attempts",
            "Abnormal process memory usage patterns",
            "Unusual sequence of system calls"
        ],
        'mitigation': [
            "Implement stack canaries and ASLR",
            "Use memory-safe languages for critical services",
            "Regularly patch vulnerable services",
            "Implement input validation and length checking"
        ]
    },
    'sql_injection': {
        'signatures': [
            "SQL keywords in URIs or form data (SELECT, UNION, etc.)",
            "Unusual quote patterns in input strings",
            "Known SQLi attack patterns",
            "Database error messages in responses"
        ],
        'anomalies': [
            "Abnormally long query strings",
            "Unusual timing delays in responses",
            "Multiple similar requests with small variations",
            "Unexpected database errors"
        ],
        'mitigation': [
            "Use parameterized queries/prepared statements",
            "Implement WAF with SQLi rules",
            "Validate and sanitize all user inputs",
            "Minimize database error messages to users"
        ]
    },
    'default': {
        'signatures': [
            "Behavior deviating from established baselines",
            "Known malicious IP addresses",
            "Patterns matching attack signatures",
            "Blacklisted domain names or URLs"
        ],
        'anomalies': [
            "Unusual traffic volume or patterns",
            "Abnormal protocol usage",
            "Suspicious timing of activities",
            "Geographic anomalies in access patterns"
        ],
        'mitigation': [
            "Implement defense in depth",
            "Regularly update IDS/IPS signatures",
            "Monitor and analyze logs systematically",
            "Follow principle of least privilege"
        ]
    }
}

# System prompt for the LLM
system_prompt = """
You are a cybersecurity expert specializing in intrusion detection systems (IDS/IPS) with deep knowledge of machine learning pipelines and attack detection methods.

When answering about the code:
1. Explain the overall architecture and data flow
2. Describe each component's purpose
3. Explain how models make decisions
4. Discuss feature engineering and preprocessing
5. Mention evaluation metrics and thresholds

For security questions:
1. Be technical but clear
2. Reference specific network features when relevant
3. For block explanations, mention confidence scores
4. Include both signature and anomaly-based detection methods
5. Provide practical mitigation strategies
6. If you don't know, say so
"""

# Feature preparation functions
def prepare_features(features_dict):
    """Convert input dictionary to properly formatted DataFrame"""
    df = pd.DataFrame([features_dict])
    
    # Default features (simplified for demo)
    default_features = {
        'duration': 0.0,
        'protocol_type': 'tcp',
        'service': 'http',
        'flag': 'SF',
        'src_bytes': 0,
        'dst_bytes': 0,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 0,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': 0,
        'srv_count': 0,
        'serror_rate': 0.0,
        'srv_serror_rate': 0.0,
        'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0,
        'same_srv_rate': 0.0,
        'diff_srv_rate': 0.0,
        'srv_diff_host_rate': 0.0,
        'dst_host_count': 0,
        'dst_host_srv_count': 0,
        'dst_host_same_srv_rate': 0.0,
        'dst_host_diff_srv_rate': 0.0,
        'dst_host_same_src_port_rate': 0.0,
        'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0,
        'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0
    }
    
    # Fill missing features with defaults
    for feature, default_value in default_features.items():
        if feature not in df.columns:
            df[feature] = default_value
    
    return df

def preprocess_features(df):
    """Mock preprocessing - in real app this would use actual scaler/encoders"""
    return df.values  # Simplified for demo

# Enhanced IDS Explanation Generator with Detection Methods
def generate_ids_explanation(ip_address):
    """Generate detailed explanation using mock model pipeline"""
    mock_connection = {
        'src_ip': ip_address,
        'duration': 0.02,
        'protocol_type': 'tcp',
        'service': 'http',
        'flag': 'SF',
        'src_bytes': 1024,
        'dst_bytes': 0,
        'logged_in': 1
    }
    
    try:
        # Mock model predictions
        attack_type = "neptune"
        attack_confidence = 0.98
        attack_type_confidence = 0.95
        threshold = ATTACK_THRESHOLDS.get(attack_type, ATTACK_THRESHOLDS['default'])
        risk_level = RISK_LEVELS.get(attack_type, 'MEDIUM')
        
        # Get detection methods for this attack
        detection_info = ATTACK_DETECTION_METHODS.get(attack_type, ATTACK_DETECTION_METHODS['default'])
        
        explanation = f"""
        🚨 **Security Alert** 🚨
        
        **IP**: {mock_connection['src_ip']}
        **Classification**: {attack_type.upper()} 
        **Risk Level**: {risk_level}
        **Confidence**: {attack_type_confidence:.0%}
        
        🔍 **Model Decision Process**:
        - Binary classifier confidence: {attack_confidence:.0%}
        - Attack type classifier confidence: {attack_type_confidence:.0%}
        - Decision threshold: {threshold:.0%}
        
        📊 **Key Features**:
        - Protocol: {mock_connection['protocol_type'].upper()}
        - Service: {mock_connection['service']}
        - Source bytes: {mock_connection['src_bytes']}
        - Destination bytes: {mock_connection['dst_bytes']}
        - Logged in: {'Yes' if mock_connection['logged_in'] else 'No'}
        
        🕵️ **Detection Methods**:
        
        **Signature-based Detection**:
        {chr(10).join(['• ' + s for s in detection_info['signatures']])}
        
        **Anomaly-based Detection**:
        {chr(10).join(['• ' + a for a in detection_info['anomalies']])}
        
        **Mitigation Strategies**:
        {chr(10).join(['• ' + m for m in detection_info['mitigation']])}
        
        ℹ️ **System Action**: {'BLOCK' if attack_type_confidence > threshold else 'MONITOR'}
        """
        
        return explanation
    
    except Exception as e:
        return f"⚠️ Error generating explanation: {str(e)}"

# Attack Detection Explanation Function
def explain_attack_detection(attack_name):
    """Provide detailed detection methods for a specific attack"""
    attack_name = attack_name.lower().replace(" ", "_")
    detection_info = ATTACK_DETECTION_METHODS.get(attack_name, ATTACK_DETECTION_METHODS['default'])
    
    response = f"""
    🔍 **{attack_name.upper()} Attack Detection Methods**
    
    **Signature-based Indicators**:
    {chr(10).join(['• ' + s for s in detection_info['signatures']])}
    
    **Anomaly-based Indicators**:
    {chr(10).join(['• ' + a for a in detection_info['anomalies']])}
    
    **Mitigation Strategies**:
    {chr(10).join(['• ' + m for m in detection_info['mitigation']])}
    
    **Model Detection Features**:
    - Protocol type and service patterns
    - Connection duration and byte patterns
    - Login attempt patterns
    - Packet header anomalies
    - Traffic volume and timing characteristics
    """
    
    return response

# Code explanation functions
def explain_pipeline():
    """Generate explanation of the entire pipeline"""
    explanation = """
    ## IDS/IPS Pipeline Architecture
    
    ### 1. Data Flow:
    - Raw network data → Feature extraction → Preprocessing → Binary classification → Attack type classification → Action decision
    
    ### 2. Key Components:
    - **Model 1 (Binary Classifier)**: Random Forest that distinguishes normal vs attack traffic
    - **Model 2 (Multiclass Classifier)**: Identifies specific attack types (23 categories)
    - **Feature Selector**: Selects top 15 most important features for binary classification
    - **Preprocessing**: Standard scaling and categorical encoding
    
    ### 3. Decision Logic:
    - Uses confidence thresholds specific to each attack type
    - Implements risk-based actions (BLOCK, ALERT_ADMIN, MONITOR)
    - Incorporates both signature and anomaly detection methods
    
    ### 4. Attack Detection Methods:
    - Signature matching for known attack patterns
    - Anomaly detection for novel attack variants
    - Behavioral analysis for sophisticated threats
    """
    return explanation

def explain_file(filename):
    """Explain specific code files"""
    explanations = {
        "pipeline.py": """
        ### Pipeline.py - Core Detection Logic
        
        **Main Functions**:
        1. `analyze_connection()`: Main entry point that:
           - Processes input features
           - Runs binary classification
           - If attack detected, classifies attack type
           - Makes blocking decision based on risk level
        
        2. `prepare_features()`: Ensures consistent feature format
        3. `preprocess_features()`: Applies scaling and encoding
        
        **Key Features**:
        - Custom thresholds per attack type
        - Risk-based action system
        - Error handling for production robustness
        - Integrated detection method logging
        """,
        
        "model1_binary.py": """
        ### Model1_binary.py - Binary Classifier Training
        
        **Process**:
        1. Loads KDD dataset (10% sample)
        2. Creates binary labels (normal vs attack)
        3. Encodes categorical features
        4. Scales numerical features
        5. Selects top 15 features using ANOVA F-value
        6. Trains Random Forest with class balancing
        
        **Key Metrics**:
        - Achieves ~99% accuracy
        - Focuses on reducing false positives
        - Uses stratified sampling
        - Tracks detection method effectiveness
        """,
        
        "model2_multiclass.py": """
        ### Model2_multiclass.py - Attack Type Classifier
        
        **Process**:
        1. Uses preprocessed data from Model 1
        2. Evaluates multiple classifiers (SVM, Decision Tree, Random Forest)
        3. Selects best model based on weighted F1-score
        4. Uses same feature scaling as Model 1
        
        **Key Features**:
        - Handles 23 attack types + normal
        - Special handling for rare attack classes
        - Consistent preprocessing with production pipeline
        - Detailed attack signature logging
        """
    }
    return explanations.get(filename, "File explanation not available.")

# Check if query is about connection status
def is_connection_query(query):
    security_keywords = ['blocked', 'allowed', 'connection', 'why was', 'security alert']
    ip_pattern = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    return any(keyword in query.lower() for keyword in security_keywords) or re.search(ip_pattern, query)

# Extract IP from query
def extract_ip(query):
    ip_match = re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", query)
    return ip_match.group(0) if ip_match else None

# Check if query is about attack detection
def is_detection_query(query):
    detection_keywords = ['how to detect', 'detection methods', 'signs of', 'indicators of']
    return any(keyword in query.lower() for keyword in detection_keywords)

# Extract attack name from detection query
def extract_attack_name(query):
    query = query.lower()
    for keyword in ['how to detect', 'detection methods for', 'signs of', 'indicators of']:
        if keyword in query:
            return query.replace(keyword, "").strip()
    return None

# UI Layout
st.title("🛡️ IDS/IPS Security Assistant")
st.caption("Powered by Groq's Llama 3 70B and your IDS models - Now with Attack Detection Methods")

with st.sidebar:
    st.header("🔍 Quick Actions")
    
    if "sidebar_query" not in st.session_state:
        st.session_state.sidebar_query = None
    
    st.header("🛡️ Attack Detection Help")
    common_attacks = ["neptune", "smurf", "nmap", "buffer overflow", "sql injection"]
    selected_attack = st.selectbox("Get detection methods for:", common_attacks)
    
    if st.button("Explain Detection"):
        st.session_state.sidebar_query = f"How to detect {selected_attack}"
    
    st.header("📚 Code Documentation")
    
    if st.button("Explain Full Pipeline"):
        st.session_state.sidebar_query = "Explain how the entire IDS/IPS pipeline works"
    
    if st.button("Explain Model 1"):
        st.session_state.sidebar_query = "Explain model1_binary.py"
    
    if st.button("Explain Model 2"):
        st.session_state.sidebar_query = "Explain model2_multiclass.py"
    
    if st.button("Explain Pipeline.py"):
        st.session_state.sidebar_query = "Explain pipeline.py"
    
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared. How can I help with IDS/IPS questions?"}
        ]
        st.session_state.sidebar_query = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
if prompt := st.chat_input("Ask about network attacks, detection methods, or connection status..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                if st.session_state.sidebar_query:
                    prompt = st.session_state.sidebar_query
                    st.session_state.sidebar_query = None
                
                if "how does the code work" in prompt.lower() or "explain the pipeline" in prompt.lower():
                    response = explain_pipeline()
                
                elif "explain pipeline.py" in prompt.lower():
                    response = explain_file("pipeline.py")
                
                elif "explain model1" in prompt.lower():
                    response = explain_file("model1_binary.py")
                
                elif "explain model2" in prompt.lower():
                    response = explain_file("model2_multiclass.py")
                
                elif is_detection_query(prompt):
                    attack_name = extract_attack_name(prompt)
                    response = explain_attack_detection(attack_name) if attack_name else "Please specify an attack type"
                
                elif is_connection_query(prompt):
                    ip_address = extract_ip(prompt)
                    if ip_address:
                        ids_explanation = generate_ids_explanation(ip_address)
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Explain this security alert:\n{ids_explanation}"}
                        ]
                        response = get_chat_response(messages)
                    else:
                        response = "Please specify an IP address"
                
                else:
                    messages = [{"role": "system", "content": system_prompt}]
                    for msg in st.session_state.messages[-6:]:
                        messages.append({"role": msg["role"], "content": msg["content"]})
                    response = get_chat_response(messages)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Handle sidebar queries
if st.session_state.get('sidebar_query'):
    query = st.session_state.sidebar_query
    st.session_state.sidebar_query = None
    
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.spinner("Generating response..."):
        if query.endswith(".py"):
            filename = query.replace("Explain ", "").strip()
            response = explain_file(filename)
        elif "pipeline" in query.lower():
            response = explain_pipeline()
        elif "how to detect" in query.lower():
            attack_name = extract_attack_name(query)
            response = explain_attack_detection(attack_name)
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            response = get_chat_response(messages)
    
    st.session_state.messages.append({"role": "assistant", "content": response})