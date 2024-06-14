// Definición de tipos en OpenCL
typedef struct {
    float real;
    float imag;
} Complex;

__kernel void FFT(__global Complex *x, int N1)
{
    int fft_num = get_global_id(0);  // Obtener el índice global

    int i, j, m, len;
    float angle;
    Complex temp, wlen, w, u, v, next_w;

    // Reordenamiento por bit reversal
    j = 0;
    for (i = 0; i < N1; ++i)
    {
        if (i < j)
        {
            temp = x[fft_num * N1 + i];
            x[fft_num * N1 + i] = x[fft_num * N1 + j];
            x[fft_num * N1 + j] = temp;
        }
        for (m = N1 >> 1; m >= 1 && j >= m; m >>= 1)
        {
            j -= m;
        }
        j += m;
    }
}
