#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) printf("Программа требует ровно 2 процесса.\n");
        MPI_Finalize();
        return 0;
    }

    int max_exp = 20;       // 2^20 байт = 1MB
    int iterations = 100;   // количество повторений для статистики

    for (int i = 0; i <= max_exp; i++) {
        int num_bytes = 1 << i;
        char *buffer = (char *)malloc(num_bytes);
        if (!buffer) MPI_Abort(MPI_COMM_WORLD, 1);

        double total_time = 0.0;

        for (int k = 0; k < iterations; k++) {
            if (rank == 0) {
                for (int j = 0; j < num_bytes; j++) buffer[j] = j % 256;

                double start = MPI_Wtime();
                MPI_Send(buffer, num_bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer, num_bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                double end = MPI_Wtime();
                total_time += (end - start);
            } else if (rank == 1) {
                MPI_Recv(buffer, num_bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer, num_bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }

        if (rank == 0) {
            double avg_time = total_time / iterations;
            double bandwidth = (num_bytes / 1e6) / avg_time; // MB/s
            printf("%d %f %f\n", num_bytes, avg_time, bandwidth);
        }

        free(buffer);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
