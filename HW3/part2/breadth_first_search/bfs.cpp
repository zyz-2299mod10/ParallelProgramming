#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <vector>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    std::vector<int> thread_local_forntier[omp_get_max_threads()];
    int new_distance = distances[frontier->vertices[0]] + 1;
    #pragma omp parallel for shared(frontier, new_frontier, distances)
    for (int i = 0; i < frontier->count; i++)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                distances[outgoing] = new_distance;
                thread_local_forntier[omp_get_thread_num()].push_back(outgoing);
            }
        }
    }

    int total_count = 0;
    for (int i = 0; i < omp_get_max_threads(); i++) {
        total_count += thread_local_forntier[i].size();
    }

    new_frontier->vertices = (int*) realloc(new_frontier->vertices, (new_frontier->count + total_count) * sizeof(int));

    int current_offset = new_frontier->count;
    new_frontier->count += total_count;
    for (int i = 0; i < omp_get_max_threads(); i++) {
        for (int outgoing : thread_local_forntier[i]) {
            new_frontier->vertices[current_offset++] = outgoing;
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
    free(frontier->vertices);
    free(new_frontier->vertices);
}

// -----------------bottom_up---------------------//
bool process_node(Graph graph, int node, int current_depth, int next_depth, solution *sol) {
    int start_edge = graph->incoming_starts[node];
    int end_edge = (node == graph->num_nodes - 1) ? graph->num_edges : graph->incoming_starts[node + 1];

    for (int edge = start_edge; edge < end_edge; edge++) {
        int incoming_neighbor = graph->incoming_edges[edge];
        if (sol->distances[incoming_neighbor] == current_depth) {
            sol->distances[node] = next_depth;
            return false; 
        }
    }
    return true;
}

bool bottom_up_step(Graph graph, int current_depth, solution *sol) {
    bool all_done = true;
    int next_depth = current_depth + 1;

    #pragma omp parallel for schedule(dynamic, 1024) reduction(& : all_done)
    for (int node = 0; node < graph->num_nodes; node++) {
        if (sol->distances[node] == NOT_VISITED_MARKER) {
            bool node_done = process_node(graph, node, current_depth, next_depth, sol);
            all_done &= node_done; 
        }
    }

    return all_done;
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    #pragma omp parallel for
    for (int node = 0; node < graph->num_nodes; node++) {
        sol->distances[node] = NOT_VISITED_MARKER;
    }
    sol->distances[ROOT_NODE_ID] = 0;

    int depth = 0;
    bool traversal_complete = false;

    while (!traversal_complete) {
        traversal_complete = bottom_up_step(graph, depth, sol);
        depth++;
    }
}

// -----------------hybrid---------------------//
void initialize_solution(Graph graph, solution *sol) {
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    sol->distances[ROOT_NODE_ID] = 0;
}

void execute_top_down(Graph graph, vertex_set *&frontier, vertex_set *&new_frontier, solution *sol, int &index, int &count) {
    vertex_set_clear(new_frontier);
    top_down_step(graph, frontier, new_frontier, sol->distances);
    
    vertex_set *tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;

    index++;
    count += frontier->count;
}

bool execute_bottom_up(Graph graph, solution *sol, int &index) {
    bool done = false;
    done = bottom_up_step(graph, index, sol);
    index++;
    return done;
}

void bfs_hybrid(Graph graph, solution *sol) {
    int hybrid_threshold = graph->num_nodes / 2; 

    vertex_set list1, list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    initialize_solution(graph, sol);

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;

    int index = 0;
    int count = 0;

    while (count < hybrid_threshold && frontier->count > 0) {
        execute_top_down(graph, frontier, new_frontier, sol, index, count);
    }

    bool traversal_complete = false;
    while (!traversal_complete) {
        traversal_complete = execute_bottom_up(graph, sol, index);
    }

    free(frontier->vertices);
    free(new_frontier->vertices);
}