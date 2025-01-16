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

// Function to perform accumulation 
void accumulate_hits_with_mpi_win(int rank, long long int local_hits, long long int tosses, double &pi_result) {
    MPI_Win win;
    long long int *global_hits = NULL;

    if (rank == 0) {
        // Root process allocates shared memory region
        MPI_Alloc_mem(sizeof(long long int), MPI_INFO_NULL, &global_hits);
        *global_hits = 0;
        MPI_Win_create(global_hits, sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    } else {
        // Worker processes create empty MPI window
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Accumulate(&local_hits, 1, MPI_LONG_LONG_INT, 0, 0, 1, MPI_LONG_LONG_INT, MPI_SUM, win);
        MPI_Win_unlock(0, win);
    }

    // Synchronize all processes
    MPI_Win_fence(0, win);

    if (rank == 0) {
        *global_hits += local_hits; // Ensure root's local hits are included
        pi_result = 4.0 * (*global_hits / (double)tosses);
        MPI_Free_mem(global_hits);
    }

    MPI_Win_free(&win);
}

int main(int argc, char **argv) {
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    long long int local_hits = estimate_pi(world_rank, world_size, tosses);

    // accumulate global number of hits
    accumulate_hits_with_mpi_win(world_rank, local_hits, tosses, pi_result);

    if (world_rank == 0) {
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}