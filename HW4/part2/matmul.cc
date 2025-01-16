#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <fstream>

using namespace std;

// Function to construct matrices by reading dimensions and data from input file
void construct_matrices(std::ifstream &in, int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr) {
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    
    if (process_rank == 0) {
        // Read dimensions of matrices
        in >> *n_ptr >> *m_ptr >> *l_ptr;
        int rows_a = *n_ptr, cols_a = *m_ptr, cols_b = *l_ptr;

        // Initialize
        *a_mat_ptr = (int*) malloc(sizeof(int) * rows_a * cols_a);
        *b_mat_ptr = (int*) malloc(sizeof(int) * cols_a * cols_b);

        // Populate matrix A 
        for (int i = 0; i < rows_a; i++) {
            for (int j = 0; j < cols_a; j++) {
                in >> (*a_mat_ptr)[i * cols_a + j];
            }
        }

        // Populate matrix B 
        for (int i = 0; i < cols_a; i++) {
            for (int j = 0; j < cols_b; j++) {
                in >> (*b_mat_ptr)[i * cols_b + j];
            }
        }
    }
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat) {
    int process_rank, process_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int num_workers = process_count - 1;
    int avg_rows = n / num_workers;
    int extra_rows = n % num_workers;
    int offset = 0, rows_assigned = 0;

    if (process_rank == 0) {
        int *result_matrix = (int*) malloc(sizeof(int) * n * l);

        // Distribute workloads to worker 
        for (int dest = 1; dest <= num_workers; dest++) {
            rows_assigned = (dest <= extra_rows) ? avg_rows + 1 : avg_rows;

            MPI_Send(&m, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(&l, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(&offset, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(&rows_assigned, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(&a_mat[offset * m], rows_assigned * m, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(&b_mat[0], m * l, MPI_INT, dest, 0, MPI_COMM_WORLD);

            offset += rows_assigned;
        }

        // Collect results 
        for (int src = 1; src <= num_workers; src++) {
            MPI_Recv(&offset, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&rows_assigned, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&result_matrix[offset * l], rows_assigned * l, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Print result 
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < l; j++) {
                printf("%d ", result_matrix[i * l + j]);
            }
            printf("\n");
        }

        // Free memory
        free(result_matrix);
    } else {
        int local_cols_a, local_cols_b;
        MPI_Recv(&local_cols_a, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&local_cols_b, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Recv(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rows_assigned, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Allocate memory for local computation
        int *local_a = (int*) malloc(sizeof(int) * rows_assigned * local_cols_a);
        int *local_b = (int*) malloc(sizeof(int) * local_cols_a * local_cols_b);
        int *local_result = (int*) malloc(sizeof(int) * rows_assigned * local_cols_b);

        MPI_Recv(local_a, rows_assigned * local_cols_a, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_b, local_cols_a * local_cols_b, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // matrix multiplication
        for (int i = 0; i < rows_assigned; i++) {
            for (int j = 0; j < local_cols_b; j++) {
                local_result[i * local_cols_b + j] = 0;
                for (int k = 0; k < local_cols_a; k++) {
                    local_result[i * local_cols_b + j] += local_a[i * local_cols_a + k] * local_b[k * local_cols_b + j];
                }
            }
        }

        // Send results back to the root process
        MPI_Send(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&rows_assigned, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_result, rows_assigned * local_cols_b, MPI_INT, 0, 0, MPI_COMM_WORLD);

        // Free memory
        free(local_a);
        free(local_b);
        free(local_result);
    }
}

void destruct_matrices(int *a_mat, int *b_mat) {
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    if (process_rank == 0) {
        free(a_mat);
        free(b_mat);
    }
}
