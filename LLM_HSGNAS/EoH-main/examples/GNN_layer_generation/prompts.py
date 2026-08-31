class GetPrompts():
    def __init__(self):
        self.prompt_task = (
            "You are a top-tier AI researcher specializing in Hardware-Aware Neural Architecture Search (HW-NAS). "
            "The essence of spectral graph neural network is to capture the information of the graph by converting the graph signal to the frequency domain and propagating and filtering the information among nodes through adaptive learning some types and parameters of filters. "
            "It seems that different types of graphs lend themselves to different filters for inter-node communication. "
            "I need help in designing a novel, hardware-friendly spectral graph neural network layer that can effectively learn node embeddings for 'citation network graphs', with the ultimate goal of improving the accuracy of node classification.\n\n"

            "Furthermore, we are deploying these GNNs on a physical edge accelerator (Xilinx Kria KV260 FPGA). "
            "The architecture MUST achieve the best Pareto front trade-off among four physical metrics:\n"
            "1. Maximize Accuracy (ACC)\n"
            "2. Minimize Single Inference Latency (LATENCY_S)\n"
            "3. Maximize Throughput (THROUGHPUT_NODES_PER_S)\n"
            "4. Minimize Energy Consumption (ENERGY_J)\n\n"

            "CRITICAL HARDWARE CONSTRAINTS FOR CODE GENERATION:\n"
            "- DO NOT inherit from PyG's `MessagePassing` base class. Its dynamic graph unrolling is hostile to ONNX export and FPGA deployment.\n"
            "- ABSOLUTELY FORBIDDEN: NEVER use `torch.sparse`, `torch.sparse_coo_tensor`, or `to_sparse()`. ONNX opset 16 DOES NOT SUPPORT sparse tensors.\n"
            "- NO BOOLEAN INDEXING: NEVER use boolean masks to index tensors (e.g., `x[mask]`). ONNX will produce an incorrect graph. Instead, use mathematical masking like `x * mask.float()` or `torch.where(mask, x, 0.0)`.\n"
            "- REQUIRED AGGREGATION METHOD: You MUST use `torch.scatter_add_` for message passing.\n"
            "- NO DYNAMIC CONTROL FLOW OR LOOPS: NEVER use `if` statements that depend on tensor values. NEVER use `for` or `while` loops to iterate over nodes or edges. Use pure, fully vectorized tensor operations.\n\n"

            "- STRICT ONNX TRACING RULES: NEVER extract dynamic tensor dimensions (e.g., `x.size(0)`, `row.size(0)`, or `.shape`) to initialize new tensors via `torch.zeros()`, `torch.ones()`, or `torch.empty()`. This strictly breaks ONNX hardware tracing. You MUST ALWAYS use `torch.zeros_like(reference_tensor)` or `torch.ones_like(reference_tensor)` instead.\n"
            "  [BAD EXAMPLE]: `torch.ones(row.size(0))` or `num_nodes = x.size(0); torch.zeros(num_nodes)`\n"
            "  [GOOD EXAMPLE]: `torch.ones_like(col)` or `torch.zeros_like(x[:, 0])`\n\n"

            "- PROJECT BEFORE PROPAGATE (CRITICAL FOR LATENCY): You MUST apply a neural network Linear projection layer (e.g., `self.lin(x)`) to reduce the high-dimensional raw node features `x` into a low-dimensional hidden space BEFORE performing any graph message passing or `scatter_add_` loops. NEVER perform `scatter_add_` on the raw input `x`, as this causes extreme memory bandwidth bottlenecks on edge devices. Always propagate on the compressed hidden features.\n\n"

            "- ERROR RECOVERY PROTOCOL: If you are provided with a 'CRITICAL ERROR' from previous evaluations, it means your previous architecture crashed during ONNX tracing or hardware deployment. You MUST meticulously analyze the error message and rewrite the code to strictly bypass the issue.\n\n"

            "Tips: There may be some heterophilic graphs that do not follow the homogeneity assumption, heterophily is more like the feature or label difference between the neighbors under the nodes with the same type!"
        )

        self.prompt_class_name = "GNN_Layer"
        self.prompt_class_initial_inputs = ['input_channels', 'output_channels']
        self.prompt_forword_func_inputs = ['x', 'x_raw', 'edge_index']
        self.prompt_forword_func_outputs = ['hidden']
        self.prompt_inout_inf = "'x' is the node's processed features, shape: [node_num, input_channels]; 'x_raw' is the node's initial features, shape: [node_num, input_channels]; 'hidden' is the learned node feature, shape: [node_num, output_channels]; and 'edge_index' includes the edge information about the graph."
        self.prompt_other_inf = "All are torch tensor."

    def get_task(self):
        return self.prompt_task

    def get_class_name(self):
        return self.prompt_class_name

    def get_class_init_inputs(self):
        return self.prompt_class_initial_inputs

    def get_forward_func_inputs(self):
        return self.prompt_forword_func_inputs

    def get_forward_func_outputs(self):
        return self.prompt_forword_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf