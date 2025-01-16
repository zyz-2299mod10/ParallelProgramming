#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

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
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    srand(time(NULL) * world_rank);
    long long int number_in_circle = 0;
    if (world_rank > 0)
    {
        // TODO: MPI workers
        number_in_circle = estimate_pi(world_rank, world_size, tosses);
        MPI_Send(&number_in_circle, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        MPI_Request requests[15];
        long long int tmp_recv[15];

        // non-blocking receive
        for(int i = 1; i < world_size; i++){
            MPI_Irecv(&(tmp_recv[i - 1]), 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &(requests[i - 1]));
        }

        number_in_circle = estimate_pi(world_rank, world_size, tosses);
        MPI_Waitall(world_size - 1, requests, MPI_STATUS_IGNORE);

        // add all results
        for(int i = 1; i < world_size; i++){
            number_in_circle += tmp_recv[i - 1];
        }
    }

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
