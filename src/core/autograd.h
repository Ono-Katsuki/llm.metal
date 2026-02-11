#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"

#define AUTOGRAD_MAX_INPUTS 4
#define AUTOGRAD_MAX_NODES  8192

typedef struct GraphNode GraphNode;

// Backward function type: given the node, compute gradients for inputs
typedef void (*BackwardFn)(GraphNode *node);

// Cleanup function for saved_data (frees tensors stored for backward)
typedef void (*CleanupFn)(void *saved_data);

struct GraphNode {
    Tensor *output;
    Tensor *inputs[AUTOGRAD_MAX_INPUTS];
    int     n_inputs;
    BackwardFn backward_fn;
    CleanupFn  cleanup_fn;       // custom destructor for saved_data
    int     checkpoint;          // 1 = recompute activation on backward
    void   *saved_data;          // arbitrary data for backward (e.g., pre-activation)
    int     visited;             // for topological sort
    int     id;
};

typedef struct {
    GraphNode nodes[AUTOGRAD_MAX_NODES];
    int       n_nodes;
} ComputeGraph;

// Create / destroy graph
ComputeGraph *graph_create(void);
void          graph_destroy(ComputeGraph *g);
void          graph_reset(ComputeGraph *g);

// Add a node to the computation graph
GraphNode *graph_add_node(ComputeGraph *g, Tensor *output,
                          Tensor **inputs, int n_inputs,
                          BackwardFn backward_fn);

// Run backward pass from a given tensor (caller must set loss->grad before calling)
void graph_backward(ComputeGraph *g, Tensor *loss);

// Zero all gradients in the graph
void graph_zero_grad(ComputeGraph *g);

#endif // AUTOGRAD_H
