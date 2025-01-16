#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstring>

long long int estimate_pi(int rank, int size, long long int tosses){
    long long int chunk = tosses / size;
    long long int start = chunk * rank;
    long long int end = (rank == size - 1) ? tosses : start + chunk;
    
    long long int local_count = 0;
    unsigned seed = time(NULL) * rank;

    for(long long int i = start; i < end; i++){
        double x = 2 * (((double) rand_r(&seed)) / RAND_MAX) - 1;
        double y = 2 * (((double) rand_r(&seed)) / RAND_MAX) - 1;
        double distance_squared = x * x + y * y;

        if (distance_squared <= 1) ++local_count; 
    }

    return local_count;
}

void initialize_max_depth(int *depth_array, int total_processes) {
    // initalize depth_array to 0
    memset(depth_array, 0, sizeof(int) * total_processes);
    
    int max_tree_depth = (int)log2(total_processes);

    for (int current_depth = 0; current_depth < max_tree_depth; ++current_depth) {
        int step_size = (int)pow(2, current_depth);

        for (int process_id = 0; process_id < total_processes; process_id += step_size) {
            depth_array[process_id]++;
        }
    }
}

void perform_binary_tree_reduction(int rank, int depth_array[], long long int &local_count) {
    long long int received_value = 0;
    int virtual_rank = rank;

    int max_depth = depth_array[0];

    for (int current_depth = 0; current_depth < max_depth; ++current_depth, virtual_rank /= 2) {
        // make sure all process in the same layer
        MPI_Barrier(MPI_COMM_WORLD);

        // check layer reduction
        if (current_depth < depth_array[rank]) {
            // recive
            if (virtual_rank % 2 == 0) {
                int source_rank = pow(2, current_depth) * (virtual_rank + 1);
                MPI_Recv(&received_value, 1, MPI_LONG_LONG, source_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                local_count += received_value;
            }
            // sender
            else {
                int target_rank = pow(2, current_depth) * (virtual_rank - 1);
                MPI_Send(&local_count, 1, MPI_LONG_LONG, target_rank, 0, MPI_COMM_WORLD);
            }
        }
    }
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);	/* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);	/* get number of processes */

    srand(time(NULL) * world_rank);
    long long int number_in_circle = estimate_pi(world_rank, world_size, tosses);

    // initialize max depth for each process
    int max_depth[16];
    initialize_max_depth(max_depth, world_size);

    // TODO: binary tree redunction
    perform_binary_tree_reduction(world_rank, max_depth, number_in_circle);

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4 * number_in_circle / ((double) tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
