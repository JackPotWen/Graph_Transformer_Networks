import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import torch
from scipy.sparse import csr_matrix, csc_matrix
import os

# 设置OpenMP环境变量以避免冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class NodeTypeAnalyzer:
    def __init__(self):
        # DBLP节点类型范围
        self.author_range = (0, 4056)
        self.paper_range = (4057, 18384)
        self.conf_range = (18385, 18404)
        
        # 节点类型名称
        self.type_names = {
            'author': 'Author Nodes',
            'paper': 'Paper Nodes',
            'conference': 'Conference Nodes'
        }
        
        # 创建结果保存目录
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
    
    def random_walk(self, G, start_node, walk_length=10, num_walks=5):
        """Perform multiple random walks for a single node"""
        walks = []
        for _ in range(num_walks):
            walk = [start_node]
            current_node = start_node
            for _ in range(walk_length - 1):
                if len(list(G.neighbors(current_node))) == 0:
                    break
                current_node = np.random.choice(list(G.neighbors(current_node)))
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
        plt.title('Node Type Distribution in RW Sampling')
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
        plt.title('Node Type Bias Analysis in RW Sampling')
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
    
    def run_analysis(self, edges, walk_length=10, num_walks=5):
        """Run complete analysis pipeline"""
        print("Building heterogeneous graph...")
        # Build heterogeneous graph
        G = nx.Graph()
        for edge_type, edge_matrix in enumerate(edges):
            indices = edge_matrix.nonzero()
            for i, j in zip(indices[0], indices[1]):
                G.add_edge(int(i), int(j))
        
        print("Performing random walk sampling...")
        # Perform random walks for all nodes
        all_walks = []
        for node in tqdm(range(G.number_of_nodes())):
            walks = self.random_walk(G, node, walk_length, num_walks)
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
            'total_walks': len(all_walks)
        }
        
        with open(os.path.join(self.result_dir, 'rw_stats.pkl'), 'wb') as f:
            pickle.dump(stats, f)
        
        # Visualize results
        self.visualize_distribution(type_ratios, 
                                  os.path.join(self.result_dir, 'distribution.png'))
        self.visualize_bias(type_counts, 
                           os.path.join(self.result_dir, 'view_bias.png'))
        
        # Print statistics
        print("\nStatistics:")
        print(f"Total walks: {len(all_walks)}")
        print("\nNode Type Distribution:")
        for node_type, count in type_counts.items():
            print(f"{self.type_names[node_type]}: {count} times ({type_ratios[node_type]*100:.1f}%)")

def main():
    # Read data
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "DBLP", "edges.pkl")
    
    print(f"Reading data from: {data_path}")
    with open(data_path, 'rb') as f:
        edges = pickle.load(f)
    
    # Create analyzer and run analysis
    analyzer = NodeTypeAnalyzer()
    analyzer.run_analysis(edges, walk_length=10, num_walks=5)

if __name__ == "__main__":
    main() 