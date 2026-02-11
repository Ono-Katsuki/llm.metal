#include "autograd.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

ComputeGraph *graph_create(void) {
    ComputeGraph *g = calloc(1, sizeof(ComputeGraph));
    return g;
}

// Free saved_data in a node using cleanup_fn if set, else plain free
static void node_cleanup_saved(GraphNode *node) {
    if (!node->saved_data) return;
    if (node->cleanup_fn) {
        node->cleanup_fn(node->saved_data);
    } else {
        free(node->saved_data);
    }
    node->saved_data = NULL;
}

void graph_destroy(ComputeGraph *g) {
    if (!g) return;
    for (int i = 0; i < g->n_nodes; i++) {
        node_cleanup_saved(&g->nodes[i]);
    }
    free(g);
}

void graph_reset(ComputeGraph *g) {
    // Free intermediate tensors (outputs of graph nodes) and saved_data
    for (int i = 0; i < g->n_nodes; i++) {
        node_cleanup_saved(&g->nodes[i]);
        // Free intermediate output tensors created during forward pass
        if (g->nodes[i].output && g->nodes[i].output->owns_data) {
            tensor_free(g->nodes[i].output);
            g->nodes[i].output = NULL;
        }
    }
    g->n_nodes = 0;
}

GraphNode *graph_add_node(ComputeGraph *g, Tensor *output,
                          Tensor **inputs, int n_inputs,
                          BackwardFn backward_fn) {
    assert(g->n_nodes < AUTOGRAD_MAX_NODES);
    assert(n_inputs <= AUTOGRAD_MAX_INPUTS);

    GraphNode *node = &g->nodes[g->n_nodes];
    memset(node, 0, sizeof(GraphNode));
    node->output = output;
    node->n_inputs = n_inputs;
    node->backward_fn = backward_fn;
    node->id = g->n_nodes;
    for (int i = 0; i < n_inputs; i++) {
        node->inputs[i] = inputs[i];
    }
    g->n_nodes++;
    return node;
}

// Find node that produces tensor t
static GraphNode *find_node_for_tensor(ComputeGraph *g, Tensor *t) {
    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i].output == t) return &g->nodes[i];
    }
    return NULL;
}

// Topological sort via DFS
static void topo_dfs(ComputeGraph *g, GraphNode *node,
                     GraphNode **order, int *count) {
    if (node->visited) return;
    node->visited = 1;

    // Visit inputs
    for (int i = 0; i < node->n_inputs; i++) {
        GraphNode *dep = find_node_for_tensor(g, node->inputs[i]);
        if (dep) topo_dfs(g, dep, order, count);
    }
    order[(*count)++] = node;
}

void graph_backward(ComputeGraph *g, Tensor *loss) {
    // Reset visited flags
    for (int i = 0; i < g->n_nodes; i++) g->nodes[i].visited = 0;

    // Build topological order
    GraphNode *order[AUTOGRAD_MAX_NODES];
    int count = 0;

    GraphNode *loss_node = find_node_for_tensor(g, loss);
    if (!loss_node) {
        fprintf(stderr, "[Autograd] Loss tensor not found in graph\n");
        return;
    }
    topo_dfs(g, loss_node, order, &count);

    // NOTE: Caller must set loss->grad before calling graph_backward.
    // For standard scalar loss, caller sets grad to 1.0.
    // For manual loss (softmax CE), caller sets grad to dlogits.

    // Reverse order for backward
    for (int i = count - 1; i >= 0; i--) {
        GraphNode *node = order[i];
        if (node->backward_fn && node->output->grad) {
            node->backward_fn(node);
        }
    }
}

void graph_zero_grad(ComputeGraph *g) {
    for (int i = 0; i < g->n_nodes; i++) {
        Tensor *out = g->nodes[i].output;
        if (out && out->grad) {
            tensor_fill(out->grad, 0.0f);
        }
        for (int j = 0; j < g->nodes[i].n_inputs; j++) {
            Tensor *inp = g->nodes[i].inputs[j];
            if (inp && inp->grad) {
                tensor_fill(inp->grad, 0.0f);
            }
        }
    }
}
