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
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);	/* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);	/* get number of processes */

    srand(time(NULL) * world_rank);
    long long int number_in_circle = estimate_pi(world_rank, world_size, tosses);

    // TODO: use MPI_Gather
    long long int tmp_recv[16];
    MPI_Gather(&number_in_circle, 1, MPI_LONG_LONG, tmp_recv, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // TODO: PI result
        number_in_circle = 0;
        for(int i = 0; i < world_size; i++){
            number_in_circle += tmp_recv[i];
        }
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
