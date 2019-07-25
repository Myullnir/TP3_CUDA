
// https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iterator>
#include <algorithm>

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

cudaError_t addWithCuda(int *c, int *a, int *b, unsigned int size);

// __global__ se usa para declarar la funcion que va a correr en la placa de video
__global__ void addKernel(int *c,  int *a, int *b, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n)
    c[index] = a[index] + b[index];
}

#define N (2048*2048)
#define THREADS_PER_BLOCK 512
// esta funcion corre en el cpu y orquesta a la gpu 
int main()
{
	// copias en host de las variables,
	// notese que no tienen dev_ delante
    unsigned int arraySize = N * sizeof(int);
	int *a,*b,*c;
	srand(time(NULL));
	// Array no populado
	a = (int *)malloc(arraySize); 
	// Array no populado
	b = (int *)malloc(arraySize); 
	c = (int *)malloc(arraySize);

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	using namespace std;
	copy(c,
		c + sizeof(c) / sizeof(c[0]),
		ostream_iterator<short>(cout, "\n"));

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, int *a, int *b, unsigned int size)
{
	//inicializa vectores de datos de entrada y salida
	// dev = device
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
	//Todas las operaciones te devuelven un status y siempre lo comparamos para encontrar errores
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
	// Nunca se deberia cambiar a menos que halla SLI
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
	// Asignacion de espacio en memoria paara los vectores, pero en la VRAM
	// Vector resultado
    cudaStatus = cudaMalloc((void**)&dev_c, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	// Vector sum1
    cudaStatus = cudaMalloc((void**)&dev_a, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	// Vector sum2
    cudaStatus = cudaMalloc((void**)&dev_b, size );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	// Luego de asignar el espacio en la VRAM
	// Se copia de la ram del CPU a la VRAM de la GPU los datos
	//Basicamente lo movemos del mother a la placa de video
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	// Esto es lo que inicia la funcion que se distribuye en los nucleos
	// Triple angle brackets mark a call from host code to device code
	addKernel<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> (dev_c, dev_a, dev_b, N);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	// Esto es similar a un thread.Join();
	// espera a que termine el hilo antes de continuar con el resto del codigo
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
	// Se realiza el movimiento de memoria desde GPU a CPU
	// Lo mismo de antes pero el camino inverso
    cudaStatus = cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:

	// Siempre de todos los siempres hay que limpiar la memoria de la placa de video
	// Actualmente yo tengo 8gb, estaria bueno nunca sobrepasar los 2gb o 4gb
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
