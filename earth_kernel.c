/**
 * earth_kernel_summary.c
 * 
 * A consolidated summary of a C-language Earth simulation kernel.
 * This file integrates concepts from:
 *   - Shallow water atmosphere solver
 *   - MPI domain decomposition (latitudinal bands)
 *   - Halo exchange and parallel I/O
 *   - Modular Earth system coupling
 *
 * Compile with: mpicc -fopenmp -O3 -o earth_kernel earth_kernel_summary.c -lm
 * Run with:     mpirun -np 4 ./earth_kernel
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

/* Physical Constants & Grid Parameters */
#define RADIUS      6371000.0   // Earth radius (meters)
#define OMEGA       7.292e-5    // Earth rotation rate (rad/s)
#define GRAVITY     9.806       // Gravity (m/s^2)
#define DT          900.0       // Timestep (seconds) – 15 minutes

#define NX          360         // Global zonal grid points (1° resolution)
#define NY          180         // Global meridional grid points
#define HALO        2           // Number of ghost rows for stencil

/* Global Grid Metrics (computed once per rank) */
double cos_lat[NY], sin_lat[NY];      // Trigonometric factors per latitude
double dx[NX][NY], dy[NX][NY];        // Physical grid spacings (meters)

/* Local Domain State Arrays (with halo) */
// We store fields as flat 1D arrays for contiguous memory access.
// Access pattern: index = j * nx_local + i
double *h, *u, *v;   // height, zonal wind, meridional wind
double *dhdt, *dudt, *dvdt; // tendencies for RK3

// MPI domain information
int rank, size;
int local_ny, start_j, end_j;
int nx_local, ny_local_with_halo;
MPI_Comm cart_comm;
int north_rank, south_rank;

/* Function Prototypes */
void init_mpi(int *argc, char ***argv);
void setup_grid_metrics();
void allocate_state_arrays();
void init_initial_conditions();
void halo_exchange(double *field, int nx, int ny_with_halo);
void compute_tendencies();
void rk3_step();
void write_restart(double time);
void finalize();

/* Main Simulation Loop */
int main(int argc, char *argv[]) {
    double sim_time = 0.0;
    double end_time = 86400.0 * 10; // 10 days

    init_mpi(&argc, &argv);
    setup_grid_metrics();
    allocate_state_arrays();
    init_initial_conditions();

    if (rank == 0) {
        printf("Earth Kernel started with %d MPI ranks.\n", size);
        printf("Local grid size: NX=%d, NY=%d\n", NX, local_ny);
    }

    // Main time loop
    while (sim_time < end_time) {
        rk3_step();            // Advance atmosphere by DT
        sim_time += DT;

        // Exchange halo data between ranks
        halo_exchange(h, NX, ny_local_with_halo);
        halo_exchange(u, NX+1, ny_local_with_halo);
        halo_exchange(v, NX, ny_local_with_halo+1);

        // Periodic I/O (e.g., every 6 hours)
        if ((int)(sim_time / 21600.0) != (int)((sim_time - DT) / 21600.0)) {
            write_restart(sim_time);
        }
    }

    finalize();
    return 0;
}

/* MPI Initialization & Domain Decomposition */
void init_mpi(int *argc, char ***argv) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Create 1D Cartesian topology (stripes in latitude)
    int dims[1] = {size};
    int periods[1] = {0};   // Not periodic at poles
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &cart_comm);

    // Get my coordinates and neighbors
    int coords[1];
    MPI_Cart_coords(cart_comm, rank, 1, coords);
    MPI_Cart_shift(cart_comm, 0, 1, &south_rank, &north_rank);

    // Determine local grid size
    local_ny = NY / size;
    start_j = coords[0] * local_ny;
    end_j = start_j + local_ny - 1;

    nx_local = NX;
    ny_local_with_halo = local_ny + 2 * HALO;
}

/* Grid Metrics Computation */
void setup_grid_metrics() {
    double dlat = M_PI / NY;          // Radians per latitude index
    double dlon = 2.0 * M_PI / NX;    // Radians per longitude index

    for (int j = 0; j < NY; j++) {
        double lat = -M_PI_2 + (j + 0.5) * dlat; // Cell center latitude
        cos_lat[j] = cos(lat);
        sin_lat[j] = sin(lat);
    }

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            double lat = -M_PI_2 + (j + 0.5) * dlat;
            dx[i][j] = RADIUS * cos_lat[j] * dlon;
            dy[i][j] = RADIUS * dlat;
        }
    }
}

/* Memory Allocation for State Arrays */
void allocate_state_arrays() {
    size_t size_h  = nx_local * ny_local_with_halo * sizeof(double);
    size_t size_u  = (nx_local + 1) * ny_local_with_halo * sizeof(double);
    size_t size_v  = nx_local * (ny_local_with_halo + 1) * sizeof(double);

    h   = (double*)malloc(size_h);
    u   = (double*)malloc(size_u);
    v   = (double*)malloc(size_v);
    dhdt= (double*)malloc(size_h);
    dudt= (double*)malloc(size_u);
    dvdt= (double*)malloc(size_v);

    if (!h || !u || !v || !dhdt || !dudt || !dvdt) {
        fprintf(stderr, "Rank %d: Memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

/* Initial Conditions (e.g., zonal jet with perturbation) */
void init_initial_conditions() {
    // Set all fields to zero
    for (int idx = 0; idx < nx_local * ny_local_with_halo; idx++) {
        h[idx] = 10000.0;   // Resting depth (10 km)
        dhdt[idx] = 0.0;
    }
    for (int idx = 0; idx < (nx_local+1) * ny_local_with_halo; idx++) {
        u[idx] = 0.0;
        dudt[idx] = 0.0;
    }
    for (int idx = 0; idx < nx_local * (ny_local_with_halo+1); idx++) {
        v[idx] = 0.0;
        dvdt[idx] = 0.0;
    }

    // Add a mid-latitude jet (for realism)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx_local; i++) {
        for (int jj = 0; jj < local_ny; jj++) {
            int j_global = start_j + jj;
            int j_local = jj + HALO;
            double lat = -M_PI_2 + (j_global + 0.5) * M_PI / NY;

            // Jet centered at 45° latitude
            if (fabs(lat - M_PI/4.0) < 0.2) {
                u[i * ny_local_with_halo + j_local] = 30.0 * exp(-pow((lat - M_PI/4.0)/0.1, 2));
            }
        }
    }
}

/* Halo Exchange (Ghost Cell Update) */
void halo_exchange(double *field, int nx, int ny_with_halo) {
    MPI_Request requests[4];
    int req_count = 0;
    int real_ny = ny_with_halo - 2 * HALO;

    // Send top real row to north neighbor, receive into bottom halo
    if (north_rank != MPI_PROC_NULL) {
        int send_offset = (HALO + real_ny - 1) * nx;
        int recv_offset = (HALO + real_ny) * nx;
        MPI_Isend(&field[send_offset], nx, MPI_DOUBLE, north_rank, 0,
                  cart_comm, &requests[req_count++]);
        MPI_Irecv(&field[recv_offset], nx, MPI_DOUBLE, north_rank, 0,
                  cart_comm, &requests[req_count++]);
    }

    // Send bottom real row to south neighbor, receive into top halo
    if (south_rank != MPI_PROC_NULL) {
        int send_offset = HALO * nx;
        int recv_offset = 0;
        MPI_Isend(&field[send_offset], nx, MPI_DOUBLE, south_rank, 1,
                  cart_comm, &requests[req_count++]);
        MPI_Irecv(&field[recv_offset], nx, MPI_DOUBLE, south_rank, 1,
                  cart_comm, &requests[req_count++]);
    }

    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
}

/* Tendency Computation (Shallow Water Core) */
void compute_tendencies() {
    // Zero out tendencies
    for (int idx = 0; idx < nx_local * ny_local_with_halo; idx++) dhdt[idx] = 0.0;
    for (int idx = 0; idx < (nx_local+1) * ny_local_with_halo; idx++) dudt[idx] = 0.0;
    for (int idx = 0; idx < nx_local * (ny_local_with_halo+1); idx++) dvdt[idx] = 0.0;

    // Compute relative vorticity at cell corners
    double *zeta = (double*)malloc((nx_local+1) * (ny_local_with_halo+1) * sizeof(double));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= nx_local; i++) {
        for (int j = 0; j <= ny_local_with_halo; j++) {
            int im1 = (i - 1 + nx_local) % nx_local;
            int jm1 = (j - 1 + ny_local_with_halo) % ny_local_with_halo;
            double dvdx = (v[i * (ny_local_with_halo+1) + j] - v[im1 * (ny_local_with_halo+1) + j]) / dx[i % NX][start_j + j - HALO];
            double dudy = (u[i * ny_local_with_halo + j] - u[i * ny_local_with_halo + jm1]) / dy[i % NX][start_j + j - HALO];
            zeta[i * (ny_local_with_halo+1) + j] = dvdx - dudy;
        }
    }

    // Compute tendencies for h, u, v
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx_local; i++) {
        for (int jj = 0; jj < local_ny; jj++) {
            int j = jj + HALO; // local index with halo
            int j_global = start_j + jj;

            // ... (detailed shallow water tendency code from earlier discussion) ...
            // For brevity, we place a placeholder update.
            // In a full implementation, this would include pressure gradient,
            // Coriolis, and mass flux divergence terms.

            dhdt[i * ny_local_with_halo + j] = 0.0; // placeholder
        }
    }

    free(zeta);
}

/* Third-Order Runge-Kutta Time Stepping */
void rk3_step() {
    // Simplified RK3 implementation (calls compute_tendencies three times)
    compute_tendencies();
    // ... update to intermediate states and apply final weights ...
    // (Full RK3 as shown in earlier detailed code)
}

/* Parallel I/O: Write Restart File (MPI-IO) */
void write_restart(double time) {
    MPI_File fh;
    char filename[256];
    sprintf(filename, "restart_%010.0f.dat", time);

    MPI_File_open(cart_comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);

    // Create subarray for this rank's portion (excluding halo)
    int gsizes[2] = {NX, NY};
    int lsizes[2] = {NX, local_ny};
    int starts[2] = {0, start_j};

    MPI_Datatype filetype;
    MPI_Type_create_subarray(2, gsizes, lsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);

    MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

    // Write the 'h' field (real rows only, contiguous in memory for each row)
    // Note: we need to pack the real rows without halos.
    // For simplicity, we write row by row.
    for (int jj = 0; jj < local_ny; jj++) {
        int local_j = jj + HALO;
        MPI_File_write_all(fh, &h[local_j * nx_local], nx_local,
                           MPI_DOUBLE, MPI_STATUS_IGNORE);
    }

    MPI_Type_free(&filetype);
    MPI_File_close(&fh);

    if (rank == 0) printf("Restart written at time %f\n", time);
}

/* Cleanup */
void finalize() {
    free(h); free(u); free(v);
    free(dhdt); free(dudt); free(dvdt);
    MPI_Finalize();
}