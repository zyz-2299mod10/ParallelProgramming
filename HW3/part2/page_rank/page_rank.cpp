#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int numVertices = num_nodes(g);
    double initial_prob = 1.0 / numVertices;
    double *prev_solution = (double *) malloc(sizeof(double) * numVertices);
    int *outgoing_counts = (int *) malloc(sizeof(int) * numVertices);
    bool is_converged = false;

    // Initialization: assign uniform probability and pre-compute outgoing sizes
    #pragma omp parallel for
    for (int i = 0; i < numVertices; i++) {
        solution[i] = initial_prob;
        outgoing_counts[i] = outgoing_size(g, i);
    }

    while (!is_converged) {
        double total_diff = 0;
        double dangling_contribution = 0;

        // Copy solution to sol_old for the new iteration
        memcpy(prev_solution, solution, sizeof(double) * numVertices);

        // Calculate the score contribution from nodes with no outgoing edges
        #pragma omp parallel for reduction(+:dangling_contribution)
        for (int i = 0; i < numVertices; i++) {
            if (outgoing_counts[i] == 0) {
                dangling_contribution += prev_solution[i];
            }
        }
        dangling_contribution *= (damping / numVertices);

        // Update scores for all nodes based on incoming edges
        #pragma omp parallel for reduction(+:total_diff)
        for (int i = 0; i < numVertices; i++) {
            double incoming_sum = 0;
            const Vertex* in_start = incoming_begin(g, i);
            const Vertex* in_end = incoming_end(g, i);

            for (const Vertex* v = in_start; v != in_end; ++v) {
                incoming_sum += prev_solution[*v] / outgoing_counts[*v];
            }

            solution[i] = (damping * incoming_sum) + ((1.0 - damping) / numVertices) + dangling_contribution;

            total_diff += fabs(solution[i] - prev_solution[i]);
        }

        is_converged = (total_diff < convergence);
    }

    free(prev_solution);
    free(outgoing_counts);

    /*
        For PP students: Implement the page rank algorithm here.  You
        are expected to parallelize the algorithm using openMP.  Your
        solution may need to allocate (and free) temporary arrays.

        Basic page rank pseudocode is provided below to get you started:

        // initialization: see example code above
        score_old[vi] = 1/numNodes;

        while (!converged) {

        // compute score_new[vi] for all nodes vi:
        score_new[vi] = sum over all nodes vj reachable from incoming edges
                            { score_old[vj] / number of edges leaving vj  }
        score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

        score_new[vi] += sum over all nodes v in graph with no outgoing edges
                            { damping * score_old[v] / numNodes }

        // compute how much per-node scores have changed
        // quit once algorithm has converged

        global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
        converged = (global_diff < convergence)
        }
    */
}