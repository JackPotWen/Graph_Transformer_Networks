import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import torch
from scipy.sparse import csr_matrix, csc_matrix
import glob
import argparse

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class HDataNodeTypeAnalyzer:
    def __init__(self):
        # DBLP node type ranges
        self.author_range = (0, 4056)
        self.paper_range = (4057, 18384)
        self.conf_range = (18385, 18404)
        
        # Node type names
        self.type_names = {
            'author': 'Author Nodes',
            'paper': 'Paper Nodes',
            'conference': 'Conference Nodes'
        }
        
        # Create result directory
        self.result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_result')
        os.makedirs(self.result_dir, exist_ok=True)
        
    def get_node_type(self, node_idx):
        """Determine node type based on node index"""
        if self.author_range[0] <= node_idx <= self.author_range[1]:
            return 'author'
        elif self.paper_range[0] <= node_idx <= self.paper_range[1]:
            return 'paper'
        elif self.conf_range[0] <= node_idx <= self.conf_range[1]:
            return 'conference'
        else:
            raise ValueError(f"Unknown node index: {node_idx}")
    
    def process_h_data(self, h_data):
        """Process H data format: merge channel weights and remove self-loops"""
        # h_data is a list of [channel1, channel2]
        # each channel contains (edge_index, edge_weight)
        
        # Get edge indices and weights from both channels
        edge_index1, edge_weight1 = h_data[0]
        edge_index2, edge_weight2 = h_data[1]
        
        # 检查数据类型并转换
        if isinstance(edge_index1, torch.Tensor):
            edge_index1 = edge_index1.numpy()
        if isinstance(edge_weight1, torch.Tensor):
            edge_weight1 = edge_weight1.numpy()
        if isinstance(edge_index2, torch.Tensor):
            edge_index2 = edge_index2.numpy()
        if isinstance(edge_weight2, torch.Tensor):
            edge_weight2 = edge_weight2.numpy()
        
        # Create a dictionary to store merged weights
        edge_dict = defaultdict(list)
        
        # Process first channel
        for i in range(edge_index1.shape[1]):
            src, dst = edge_index1[:, i]
            if src != dst:  # Remove self-loops
                edge_dict[(src, dst)].append(edge_weight1[i])
        
        # Process second channel
        for i in range(edge_index2.shape[1]):
            src, dst = edge_index2[:, i]
            if src != dst:  # Remove self-loops
                edge_dict[(src, dst)].append(edge_weight2[i])
        
        # Calculate average weights and create new edge list
        edges = []
        weights = []
        for (src, dst), weight_list in edge_dict.items():
            if len(weight_list) == 2:  # Only keep edges that exist in both channels
                avg_weight = sum(weight_list) / len(weight_list)
                edges.append((src, dst))
                weights.append(avg_weight)
        
        return edges, weights
    
    def build_graph(self, edges, weights):
        """Build NetworkX graph from processed edges and weights"""
        G = nx.DiGraph()  # Use directed graph since we have directed edges
        for (src, dst), weight in zip(edges, weights):
            G.add_edge(int(src), int(dst), weight=float(weight))
        return G
    
    def weighted_random_walk(self, G, start_node, walk_length=10, num_walks=5):
        """Perform multiple weighted random walks for a single node"""
        walks = []
        for _ in range(num_walks):
            walk = [start_node]
            current_node = start_node
            for _ in range(walk_length - 1):
                # Get out-edges (since we're using directed graph)
                out_edges = list(G.out_edges(current_node, data=True))
                if not out_edges:
                    break
                
                # Get weights of out-edges
                neighbors = [edge[1] for edge in out_edges]
                weights = [edge[2]['weight'] for edge in out_edges]
                
                # Convert weights to probabilities
                probs = np.array(weights) / sum(weights)
                
                # Choose next node based on weights
                current_node = np.random.choice(neighbors, p=probs)
                walk.append(current_node)
            walks.append(walk)
        return walks
    
    def analyze_walks(self, walks):
        """Analyze the distribution of node types in walk sequences"""
        type_counts = defaultdict(int)
        total_nodes = 0
        
        for walk in walks:
            for node in walk:
                node_type = self.get_node_type(node)
                type_counts[node_type] += 1
                total_nodes += 1
        
        # Calculate ratios
        type_ratios = {k: v/total_nodes for k, v in type_counts.items()}
        return type_counts, type_ratios
    
    def visualize_distribution(self, type_ratios, save_path):
        """Plot node type distribution pie chart"""
        plt.figure(figsize=(10, 8))
        labels = [self.type_names[k] for k in type_ratios.keys()]
        sizes = list(type_ratios.values())
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=(0.05, 0.05, 0.05))
        plt.axis('equal')
        plt.title('Node Type Distribution in H-Data RW Sampling')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_bias(self, type_counts, save_path):
        """Visualize node type bias"""
        plt.figure(figsize=(12, 6))
        types = list(self.type_names.values())
        counts = [type_counts[k] for k in self.type_names.keys()]
        
        # Calculate percentages
        total = sum(counts)
        actual_percentages = [count/total * 100 for count in counts]
        uniform_percentage = 100/3  # Theoretical uniform distribution (33.33%)
        uniform_dist = [uniform_percentage] * 3
        
        x = np.arange(len(types))
        width = 0.35
        
        plt.bar(x - width/2, actual_percentages, width, label='Actual Distribution', color='#66b3ff')
        plt.bar(x + width/2, uniform_dist, width, label='Uniform Distribution', color='#ff9999')
        
        plt.xlabel('Node Types')
        plt.ylabel('Percentage (%)')
        plt.title('Node Type Bias Analysis in H-Data RW Sampling')
        plt.xticks(x, types)
        plt.legend()
        
        # Add percentage labels on top of bars
        for i, v in enumerate(actual_percentages):
            plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center')
        for i, v in enumerate(uniform_dist):
            plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center')
        
        plt.ylim(0, max(max(actual_percentages), uniform_percentage) * 1.2)  # Add 20% padding
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self, h_data_path, walk_length=10, num_walks=5):
        """Run complete analysis pipeline"""
        print(f"Reading H data from: {h_data_path}")
        with open(h_data_path, 'rb') as f:
            h_data = pickle.load(f)
        
        print("Processing H data: merging channel weights and removing self-loops...")
        edges, weights = self.process_h_data(h_data)
        
        print("Building directed heterogeneous graph...")
        G = self.build_graph(edges, weights)
        
        print("Performing weighted random walk sampling...")
        # Perform random walks for all nodes
        all_walks = []
        for node in tqdm(range(G.number_of_nodes())):
            walks = self.weighted_random_walk(G, node, walk_length, num_walks)
            all_walks.extend(walks)
        
        print("Analyzing node distribution...")
        # Analyze node distribution
        type_counts, type_ratios = self.analyze_walks(all_walks)
        
        # Save statistics
        stats = {
            'type_counts': dict(type_counts),
            'type_ratios': type_ratios,
            'walk_length': walk_length,
            'num_walks': num_walks,
            'total_walks': len(all_walks),
            'total_edges': len(edges),
            'total_nodes': G.number_of_nodes()
        }
        
        # Get epoch number from filename
        epoch_num = os.path.basename(h_data_path).split('_')[-1].split('.')[0]
        stats_path = os.path.join(self.result_dir, f'h_rw_stats_epoch{epoch_num}.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        
        # Visualize results
        self.visualize_distribution(type_ratios, 
                                  os.path.join(self.result_dir, f'distribution_h_epoch{epoch_num}.png'))
        self.visualize_bias(type_counts, 
                           os.path.join(self.result_dir, f'view_bias_h_epoch{epoch_num}.png'))
        
        # Print statistics
        print("\nStatistics:")
        print(f"Total nodes: {G.number_of_nodes()}")
        print(f"Total edges: {len(edges)}")
        print(f"Total walks: {len(all_walks)}")
        print("\nNode Type Distribution:")
        for node_type, count in type_counts.items():
            print(f"{self.type_names[node_type]}: {count} times ({type_ratios[node_type]*100:.1f}%)")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Analyze node type distribution in H-data random walks')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number to analyze (e.g., 12 for metapath_H_Epoch12.pkl)')
    parser.add_argument('--walk_length', type=int, default=10, help='Length of each random walk (default: 10)')
    parser.add_argument('--num_walks', type=int, default=5, help='Number of walks per node (default: 5)')
    
    args = parser.parse_args()
    
    # Get the path to H_Epoch directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    h_data_dir = os.path.join(base_dir, "metapath_data", "H", "DBLP", "H_Epoch")
    
    # 构建指定epoch的文件路径
    h_data_file = os.path.join(h_data_dir, f"metapath_H_Epoch{args.epoch}.pkl")
    
    if not os.path.exists(h_data_file):
        raise FileNotFoundError(f"H data file not found: {h_data_file}")
    
    print(f"\nProcessing {os.path.basename(h_data_file)}...")
    analyzer = HDataNodeTypeAnalyzer()
    analyzer.run_analysis(h_data_file, walk_length=args.walk_length, num_walks=args.num_walks)

if __name__ == "__main__":
    main() 