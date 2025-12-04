"""
Bad Actor Ring Detector

This module detects coordinated scam networks using graph clustering techniques.
It identifies when multiple accounts are operating together by analyzing shared
infrastructure (devices, IPs) and behavioral patterns (similar messages/bios).
"""

import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Optional, Tuple
import ipaddress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BadActorRingDetector:
    """
    Detects coordinated scam networks using graph-based clustering.
    
    Identifies rings of bad actors by analyzing:
    - Shared device hashes
    - Shared IP addresses/subnets
    - Similar message/bio content
    """
    
    def __init__(self):
        """
        Initialize the Bad Actor Ring Detector.
        
        No parameters required. Sets up internal configurations.
        """
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    
    def build_user_graph(self, profiles: list, conversations: list = None) -> nx.Graph:
        """
        Build a network graph from user profiles and conversations.
        
        Creates a graph where nodes represent user profiles and edges represent
        shared infrastructure or behavioral similarities.
        
        Args:
            profiles: List of profile dictionaries with keys:
                - id: Unique profile identifier
                - bio: Profile bio text (optional)
                - device_hash: Device identifier hash (optional)
                - ip: IP address (optional)
                - photo_count: Number of photos (optional)
                - account_age_days: Account age in days (optional)
                - risk_score: Risk score 0-1 (optional, defaults to 0.5)
            conversations: List of conversation dictionaries (currently unused,
                reserved for future expansion)
        
        Returns:
            networkx.Graph with:
                - Nodes: User profiles with attributes (risk_score, account_age_days)
                - Edges: Connections with weight attribute representing connection strength
        """
        try:
            G = nx.Graph()
            
            if not profiles or len(profiles) == 0:
                logger.warning("Empty profiles list provided")
                return G
            
            # Add nodes with attributes
            for profile in profiles:
                profile_id = profile.get('id') or profile.get('profile_id')
                if not profile_id:
                    logger.warning(f"Profile missing ID, skipping: {profile}")
                    continue
                
                G.add_node(
                    profile_id,
                    risk_score=profile.get('risk_score', 0.5),
                    account_age_days=profile.get('account_age_days', 0),
                    bio=profile.get('bio', ''),
                    device_hash=profile.get('device_hash'),
                    ip=profile.get('ip')
                )
            
            # Build edges based on shared infrastructure and similarity
            profile_list = list(G.nodes(data=True))
            
            for i, (node1, attrs1) in enumerate(profile_list):
                for j, (node2, attrs2) in enumerate(profile_list[i+1:], start=i+1):
                    weight = 0.0
                    edge_details = {
                        'shared_device': False,
                        'shared_ip': False,
                        'similar_bio': False
                    }
                    
                    # Check shared device hash
                    device1 = attrs1.get('device_hash')
                    device2 = attrs2.get('device_hash')
                    if device1 and device2 and device1 == device2:
                        weight += 0.7
                        edge_details['shared_device'] = True
                    
                    # Check shared IP subnet
                    ip1 = attrs1.get('ip')
                    ip2 = attrs2.get('ip')
                    if ip1 and ip2 and self._same_ip_subnet(ip1, ip2):
                        weight += 0.5
                        edge_details['shared_ip'] = True
                    
                    # Check bio similarity
                    bio1 = attrs1.get('bio', '')
                    bio2 = attrs2.get('bio', '')
                    if bio1 and bio2:
                        similarity = self._message_similarity(bio1, bio2)
                        if similarity > 0.85:
                            weight += 0.6
                            edge_details['similar_bio'] = True
                    
                    # Only add edge if weight is significant
                    if weight > 0.4:
                        G.add_edge(node1, node2, weight=weight, **edge_details)
            
            logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.error(f"Error building user graph: {e}")
            return nx.Graph()
    
    def detect_rings(self, G: nx.Graph, eps: float = 0.4, min_samples: int = 2) -> List[Dict]:
        """
        Detect coordinated rings using DBSCAN clustering on the graph.
        
        Args:
            G: networkx.Graph from build_user_graph
            eps: DBSCAN epsilon parameter (maximum distance between samples)
            min_samples: Minimum samples required for a cluster
        
        Returns:
            List of ring dictionaries, each containing:
                - ring_id: Unique identifier for the ring
                - members: List of user IDs in the ring
                - size: Number of members
                - severity_score: 0-1 score based on ring cohesion
                - action: Recommended action ('SUSPEND', 'FLAG', or 'MONITOR')
                - edge_details: Dictionary with counts of connection types
        """
        try:
            if G is None or G.number_of_nodes() == 0:
                logger.warning("Empty graph provided, returning empty rings list")
                return []
            
            if G.number_of_edges() == 0:
                logger.info("Graph has no edges, no rings detected")
                return []
            
            # Convert graph to adjacency matrix
            nodes = list(G.nodes())
            adj_matrix = nx.to_numpy_array(G, nodelist=nodes, weight='weight')
            
            # Handle case where all weights are 0 or matrix is empty
            if np.all(adj_matrix == 0):
                logger.info("Adjacency matrix has no connections")
                return []
            
            # Use DBSCAN clustering
            # Convert adjacency to distance matrix (invert weights, higher weight = closer)
            # For DBSCAN, we need distance, so we'll use 1 - normalized_weight
            max_weight = np.max(adj_matrix) if np.max(adj_matrix) > 0 else 1.0
            distance_matrix = 1.0 - (adj_matrix / max_weight)
            
            # DBSCAN expects a distance matrix or feature matrix
            # We'll use the adjacency matrix directly and adjust eps
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            labels = clustering.fit_predict(distance_matrix)
            
            # Process clusters
            rings = []
            unique_labels = set(labels)
            
            # Remove noise label (-1)
            unique_labels.discard(-1)
            
            for label in unique_labels:
                # Get member node IDs for this cluster
                member_indices = np.where(labels == label)[0]
                members = [nodes[i] for i in member_indices]
                
                if len(members) < min_samples:
                    continue
                
                # Calculate severity score
                severity_score = self._calculate_ring_severity(G, members)
                
                # Count edge types
                edge_details = self._count_edge_types(G, members)
                
                # Determine action
                if severity_score > 0.85:
                    action = 'SUSPEND'
                elif severity_score > 0.70:
                    action = 'FLAG'
                else:
                    action = 'MONITOR'
                
                rings.append({
                    'ring_id': f'ring_{label}',
                    'members': members,
                    'size': len(members),
                    'severity_score': round(severity_score, 3),
                    'action': action,
                    'edge_details': edge_details
                })
            
            # Sort by severity descending
            rings.sort(key=lambda x: x['severity_score'], reverse=True)
            
            logger.info(f"Detected {len(rings)} rings")
            return rings
            
        except Exception as e:
            logger.error(f"Error detecting rings: {e}")
            return []
    
    def _calculate_ring_severity(self, G: nx.Graph, members: List[str]) -> float:
        """
        Calculate severity score for a detected ring.
        
        Args:
            G: networkx.Graph
            members: List of node IDs in the ring
        
        Returns:
            Severity score between 0.0 and 1.0
        """
        try:
            if not members or len(members) < 2:
                return 0.0
            
            # Calculate cluster density
            # Get subgraph of members
            subgraph = G.subgraph(members)
            actual_edges = subgraph.number_of_edges()
            
            # Possible edges in complete graph of n nodes = n*(n-1)/2
            n = len(members)
            possible_edges = n * (n - 1) / 2 if n > 1 else 1
            
            cluster_density = actual_edges / possible_edges if possible_edges > 0 else 0.0
            
            # Calculate average risk score
            risk_scores = [G.nodes[m].get('risk_score', 0.5) for m in members if m in G.nodes()]
            avg_risk = np.mean(risk_scores) if risk_scores else 0.5
            
            # Combined severity score
            severity = (cluster_density * 0.6) + (avg_risk * 0.4)
            
            return min(severity, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating ring severity: {e}")
            return 0.5  # Default to medium severity on error
    
    def _count_edge_types(self, G: nx.Graph, members: List[str]) -> Dict:
        """
        Count different types of connections within a ring.
        
        Args:
            G: networkx.Graph
            members: List of node IDs in the ring
        
        Returns:
            Dictionary with counts of connection types
        """
        try:
            edge_details = {
                'shared_devices': 0,
                'shared_ips': 0,
                'similar_bios': 0,
                'total_connections': 0
            }
            
            subgraph = G.subgraph(members)
            
            for u, v, data in subgraph.edges(data=True):
                edge_details['total_connections'] += 1
                if data.get('shared_device', False):
                    edge_details['shared_devices'] += 1
                if data.get('shared_ip', False):
                    edge_details['shared_ips'] += 1
                if data.get('similar_bio', False):
                    edge_details['similar_bios'] += 1
            
            return edge_details
            
        except Exception as e:
            logger.warning(f"Error counting edge types: {e}")
            return {
                'shared_devices': 0,
                'shared_ips': 0,
                'similar_bios': 0,
                'total_connections': 0
            }
    
    def _same_ip_subnet(self, ip1: str, ip2: str) -> bool:
        """
        Check if two IP addresses are in the same /24 subnet.
        
        Args:
            ip1: First IP address as string
            ip2: Second IP address as string
        
        Returns:
            True if IPs are in same /24 subnet, False otherwise
        
        Examples:
            '192.168.1.5' and '192.168.1.10' -> True
            '192.168.1.5' and '192.168.2.10' -> False
        """
        try:
            if not ip1 or not ip2:
                return False
            
            # Parse IP addresses
            ip_obj1 = ipaddress.IPv4Address(ip1)
            ip_obj2 = ipaddress.IPv4Address(ip2)
            
            # Get /24 network for each IP
            network1 = ipaddress.IPv4Network(f"{ip_obj1}/24", strict=False)
            network2 = ipaddress.IPv4Network(f"{ip_obj2}/24", strict=False)
            
            # Check if they're in the same network
            return network1 == network2
            
        except (ValueError, ipaddress.AddressValueError) as e:
            logger.warning(f"Error comparing IP subnets: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error in _same_ip_subnet: {e}")
            return False
    
    def _message_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two text strings.
        
        Uses TF-IDF vectorization to convert text to vectors, then computes
        cosine similarity between the vectors.
        
        Args:
            text1: First text string
            text2: Second text string
        
        Returns:
            Cosine similarity score between 0.0 and 1.0
        """
        try:
            if not text1 or not text2:
                return 0.0
            
            if text1 == text2:
                return 1.0
            
            # Vectorize texts
            try:
                vectors = self.vectorizer.fit_transform([text1, text2])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                return float(similarity)
            except ValueError:
                # Handle case where vectorizer fails (e.g., empty strings after processing)
                return 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating message similarity: {e}")
            return 0.0


if __name__ == "__main__":
    # Test the ring detector
    detector = BadActorRingDetector()
    
    # Test profiles with shared infrastructure
    test_profiles = [
        {
            'id': 'user1',
            'bio': 'Love traveling and meeting new people!',
            'device_hash': 'abc123',
            'ip': '192.168.1.5',
            'risk_score': 0.8,
            'account_age_days': 3
        },
        {
            'id': 'user2',
            'bio': 'Love traveling and meeting new people!',
            'device_hash': 'abc123',
            'ip': '192.168.1.10',
            'risk_score': 0.75,
            'account_age_days': 5
        },
        {
            'id': 'user3',
            'bio': 'Different bio here',
            'device_hash': 'xyz789',
            'ip': '192.168.2.5',
            'risk_score': 0.3,
            'account_age_days': 200
        }
    ]
    
    # Build graph
    G = detector.build_user_graph(test_profiles)
    print(f"Graph nodes: {G.number_of_nodes()}")
    print(f"Graph edges: {G.number_of_edges()}")
    
    # Detect rings
    rings = detector.detect_rings(G)
    print(f"\nDetected {len(rings)} rings:")
    for ring in rings:
        print(f"  {ring['ring_id']}: {ring['size']} members, severity={ring['severity_score']}, action={ring['action']}")

