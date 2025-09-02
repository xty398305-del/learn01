#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <memory>
#include <stdexcept>

using namespace std;

class Graph {
private:
    int n; // �ڵ�����
    vector<vector<int>> adj_list; // �ڽӱ�洢
    vector<int> indptr; // CSR��ʽ����ָ��
    vector<int> indices; // CSR��ʽ������
    vector<double> data; // CSR��ʽ����
    vector<double> d_inv_sqrt; // ��һ�������� D^{-1/2}
    bool normalized = false;

public:
    // ���캯������ʼ��n���ڵ�
    Graph(int num_nodes) : n(num_nodes), adj_list(num_nodes) {}

    // ��ӱ�
    void add_edge(int u, int v) {
        if (u < 0 || u >= n || v < 0 || v >= n)
            throw out_of_range("Node index out of range");

        // �����ظ����
        if (find(adj_list[u].begin(), adj_list[u].end(), v) == adj_list[u].end()) {
            adj_list[u].push_back(v);
            adj_list[v].push_back(u); // ����ͼ
            normalized = false; // ͼ�ṹ�仯����Ҫ���¹�һ��
        }
    }

    // ���ڽӱ���CSR��ʽ�������Ի���
    void build_csr() {
        if (!indptr.empty() && normalized) return; // �ѹ�����δ�仯

        indptr.clear();
        indices.clear();
        data.clear();

        indptr.push_back(0);
        for (int i = 0; i < n; i++) {
            // ����Ի�
            indices.push_back(i);
            data.push_back(1.0);

            // ����ھ�
            for (int neighbor : adj_list[i]) {
                // �����ظ�����Ի�
                if (neighbor != i) {
                    indices.push_back(neighbor);
                    data.push_back(1.0);
                }
            }
            indptr.push_back(indices.size());
        }
        normalized = false; // ��Ҫ���¼����һ��
    }

    // �����һ��������
    void compute_normalization() {
        if (normalized) return;
        build_csr();

        d_inv_sqrt.resize(n);
        for (int i = 0; i < n; i++) {
            // �ڵ���� = ���з���Ԫ�ظ���
            double degree = indptr[i+1] - indptr[i];
            d_inv_sqrt[i] = (degree > 0) ? 1.0 / sqrt(degree) : 0.0;
        }
        normalized = true;
    }

    // BFS�ж�ͼ��ͨ��
    bool is_connected() {
        if (n == 0) return true;

        vector<bool> visited(n, false);
        queue<int> q;
        int count = 0;

        q.push(0);
        visited[0] = true;

        while (!q.empty()) {
            int node = q.front();
            q.pop();
            count++;

            for (int neighbor : adj_list[node]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }

        return count == n;
    }

    // ϡ�����˷� (CSR��ʽ �� ���ܾ���)
    vector<vector<double>> sparse_mult(const vector<vector<double>>& X) {
        compute_normalization();

        int d = X[0].size();
        vector<vector<double>> result(n, vector<double>(d, 0.0));

        for (int i = 0; i < n; i++) {
            int start = indptr[i];
            int end = indptr[i + 1];

            for (int k = start; k < end; k++) {
                int j = indices[k];
                double a_ij = data[k];

                // Ӧ�ù�һ��: \bar{A}_{ij} = D^{-1/2}_ii * A_{ij} * D^{-1/2}_jj
                double norm_val = d_inv_sqrt[i] * a_ij * d_inv_sqrt[j];

                for (int col = 0; col < d; col++) {
                    result[i][col] += norm_val * X[j][col];
                }
            }
        }

        return result;
    }

    // ��ȡ�ڵ�����
    int num_nodes() const { return n; }

    // ��ӡCSR������Ϣ�����ڵ��ԣ�
    void print_csr_info() const {
        cout << "CSR Matrix Information:\n";
        cout << "indptr: ";
        for (int i : indptr) cout << i << " ";
        cout << "\nindices: ";
        for (int i : indices) cout << i << " ";
        cout << "\ndata: ";
        for (double d : data) cout << d << " ";
        cout << "\n";
    }
};

// ��ӡ����
void print_matrix(const string& title, const vector<vector<double>>& matrix) {
    cout << "\n" << title << ":\n";
    cout << "     ";
    for (size_t j = 0; j < matrix[0].size(); j++) {
        cout << "Feature " << j+1 << "    ";
    }
    cout << "\n";

    for (size_t i = 0; i < matrix.size(); i++) {
        cout << "N" << i+1 << "  ";
        for (double val : matrix[i]) {
            cout << fixed << setprecision(4) << setw(10) << val << " ";
        }
        cout << endl;
    }
    cout << endl;
}

class GCNLayer {
private:
    int input_dim;
    int output_dim;
    vector<vector<double>> weights;
    bool use_relu;
    int layer_id;

public:
    // ���캯������ʼ��Ȩ��
    GCNLayer(int in_dim, int out_dim, int id, bool relu=true)
        : input_dim(in_dim), output_dim(out_dim), use_relu(relu), layer_id(id) {

        // �����ʼ��Ȩ�أ���ֵΪ0����׼��Ϊ0.1����̬�ֲ���
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> dist(0.0, 0.1);

        weights.resize(input_dim, vector<double>(output_dim));
        for (int i = 0; i < input_dim; i++) {
            for (int j = 0; j < output_dim; j++) {
                weights[i][j] = dist(gen);
            }
        }

        // ��ӡ��ʼ����Ϣ
        cout << "Initialized GCN Layer " << layer_id << ": "
             << input_dim << " -> " << output_dim
             << (use_relu ? " with ReLU" : "") << endl;
    }

    // ReLU�����
    vector<vector<double>> relu(const vector<vector<double>>& X) {
        vector<vector<double>> result = X;
        for (auto& row : result) {
            for (double& val : row) {
                val = max(0.0, val);
            }
        }
        return result;
    }

    // ǰ�򴫲�����ӡ�м�������
    vector<vector<double>> forward(Graph& graph, const vector<vector<double>>& X, const string& stage_name) {
        // ϡ�����˷� (������һ��): X1 = \bar{A} X
        vector<vector<double>> X1 = graph.sparse_mult(X);
        print_matrix("After sparse multiplication (A*X) in " + stage_name, X1);

        // ���Ա任: X2 = X1 W
        vector<vector<double>> X2(X1.size(), vector<double>(output_dim, 0.0));
        for (size_t i = 0; i < X1.size(); i++) {
            for (int k = 0; k < input_dim; k++) {
                for (int j = 0; j < output_dim; j++) {
                    X2[i][j] += X1[i][k] * weights[k][j];
                }
            }
        }
        print_matrix("After linear transform (X*W) in " + stage_name, X2);

        // ReLU��������Ҫ��
        vector<vector<double>> result = use_relu ? relu(X2) : X2;

        if (use_relu) {
            print_matrix("After ReLU activation in " + stage_name, result);
        }

        return result;
    }
};

// ��ӡͼ�ṹ
void print_graph_structure(const Graph& graph) {
    cout << "\nGraph Structure:\n";
    cout << "Nodes: " << graph.num_nodes() << endl;
    cout << "Edges:\n";
    // ����Graph�ڲ�ʹ���ڽӱ�����򻯴�ӡ
    cout << "1-2, 1-3, 2-3, 3-4, 4-5\n";
}

class GCNModel {
private:
    vector<shared_ptr<GCNLayer>> layers;
    int layer_count = 0;

public:
    // ��Ӳ�
    void add_layer(int input_dim, int output_dim, bool relu=true) {
        layers.push_back(make_shared<GCNLayer>(input_dim, output_dim, ++layer_count, relu));
    }

    // ǰ�򴫲�����㣩
    vector<vector<double>> forward(Graph& graph, const vector<vector<double>>& X) {
        vector<vector<double>> output = X;
        print_matrix("\nInitial input features", output);

        for (size_t i = 0; i < layers.size(); i++) {
            string stage = "Layer " + to_string(i+1);
            output = layers[i]->forward(graph, output, stage);
        }

        return output;
    }
};



int main() {
    // ����ͼ��5���ڵ㣩
    Graph graph(5);

    // ��ӱߣ�����ͼ��
    graph.add_edge(0, 1); // 1-2
    graph.add_edge(0, 2); // 1-3
    graph.add_edge(1, 2); // 2-3
    graph.add_edge(2, 3); // 3-4
    graph.add_edge(3, 4); // 4-5

    // �����ͨ��
    cout << "Graph is " << (graph.is_connected() ? "connected" : "disconnected") << endl;
    print_graph_structure(graph);

    // ��ʼ���ڵ�������5���ڵ㣬ÿ���ڵ�3ά������
    vector<vector<double>> X = {
        {1.0, 0.0, 0.0}, // Node 1
        {0.0, 1.0, 0.0}, // Node 2
        {0.0, 0.0, 1.0}, // Node 3
        {1.0, 1.0, 0.0}, // Node 4
        {0.0, 1.0, 1.0}  // Node 5
    };

    // �������GCNģ��
    GCNModel model;
    cout << "\nBuilding GCN Model:\n";
    model.add_layer(3, 4);       // Layer 1: 3 input, 4 output, with ReLU
    model.add_layer(4, 4);       // Layer 2: 4 input, 4 output, with ReLU
    model.add_layer(4, 2, false); // Layer 3: 4 input, 2 output, no ReLU

    // ���ǰ�򴫲�
    cout << "\nStarting Forward Propagation:\n";
    vector<vector<double>> output = model.forward(graph, X);

    print_matrix("Final Output", output);

    return 0;
}
