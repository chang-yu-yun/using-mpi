#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

#define MIN(a, b)  (a) <= (b) ? (a) : (b)

void print_mat(double* mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.4f", mat[i*cols + j]);
        }
        printf("\n");
    }
}

void fill_mat(double* mat, int rows, int cols)
{
    srand(time(NULL));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i*cols + j] = (double)rand() / RAND_MAX;
        }
    }
}

int main(int argc, char* argv[])
{
    int rows, cols;
    if (argc != 3) {
        fprintf(stderr, "Usage: %s nrows ncols\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    rows = atoi(argv[1]);
    cols = atoi(argv[2]);

    MPI_Init(NULL, NULL);

    int manager = 0;
    int nrank, irank;
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);
    MPI_Comm_rank(MPI_COMM_WORLD, &irank);

    if (irank == manager) {
        // memory allocation
        double* mat = (double*)malloc(sizeof(double) * rows * cols);
        double* vec = (double*)malloc(sizeof(double) * cols);
        double* res = (double*)malloc(sizeof(double) * rows);
        int count = 0, sender, tag;
        double ans;
        MPI_Status status;

        // matrix initialization
        fill_mat(mat, rows, cols);
        fill_mat(vec, cols, 1);
        memset(res, 0, sizeof(double) * rows);

        // broadcast vector
        MPI_Bcast(vec, cols, MPI_DOUBLE, manager, MPI_COMM_WORLD);

        // send rows to workers
        for (int i = 0; i < MIN(rows, nrank-1); i++) {
            MPI_Send(mat + count*cols, cols, MPI_DOUBLE, i+1, count+1, MPI_COMM_WORLD);
            count++;
        }

        // receive multiplication results
        for (int i = 0; i < rows; i++) {
            MPI_Recv(&ans, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            sender = status.MPI_SOURCE;
            tag = status.MPI_TAG;
            res[tag-1] = ans;
            if (count < rows) {
                MPI_Send(mat + count*cols, cols, MPI_DOUBLE, sender, count+1, MPI_COMM_WORLD);
                count++;
            } else {
                MPI_Send(MPI_BOTTOM, 0, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD);
            }
        }

        // print matrix, vector and multiplication result
        printf("Matrix A:\n");
        print_mat(mat, rows, cols);
        printf("\nVector b:\n");
        print_mat(vec, cols, 1);
        printf("\nResult c:\n");
        print_mat(res, rows, 1);
        free(mat);
        free(vec);
        free(res);
    } else {
        // memory allocation
        double* row = (double*)malloc(sizeof(double) * cols);
        double* vec = (double*)malloc(sizeof(double) * cols);
        int row_idx;
        double res;
        MPI_Status status;

        // receive vector from manager
        MPI_Bcast(vec, cols, MPI_DOUBLE, manager, MPI_COMM_WORLD);

        // this rank has task
        if (irank <= rows) {
            for (;;) {
                // receive one row of matrix from manager
                MPI_Recv(row, cols, MPI_DOUBLE, manager, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                row_idx = status.MPI_TAG;

                // no work required to handle
                if (row_idx == 0) break;
                res = 0.0;
                for (int i = 0; i < cols; i++) {
                    res = res + row[i] * vec[i];
                }

                // send product result back to manager
                MPI_Send(&res, 1, MPI_DOUBLE, manager, row_idx, MPI_COMM_WORLD);
            }
        }
        free(row);
        free(vec);
    }

    MPI_Finalize();

    exit(EXIT_SUCCESS);
}
