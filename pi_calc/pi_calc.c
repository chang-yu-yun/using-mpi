#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
    const double PI25DT = 3.141592653589793238462643;

    MPI_Init(&argc, &argv);

    int nproc, irank, n;
    double width, height, mid, local_sum, pi;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &irank);

    for (;;) {
        if (irank == 0) {
            printf("\nEnter the number of intervals: (0 quits) ");
            scanf("%d", &n);
        }
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (n == 0) break;
        width = 1.0 / (double)n;
        local_sum = 0.0;
        for (int i = irank; i < n; i += nproc) {
            mid = ((double)i+0.5) * width;
            height = 4.0 / (1.0 + mid*mid);
            local_sum = local_sum + height*width;
        }
        MPI_Reduce(&local_sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (irank == 0) {
            printf("Pi: %f\n", pi);
            printf("Error: %f\n", fabs(pi-PI25DT));
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
