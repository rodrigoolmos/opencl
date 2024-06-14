__kernel void FFT(__global float2 *x, int N1, int N2)
{
    int fft_num = get_global_id(0);  // Obtener el índice global

    int i, j, m, len;
    float angle;
    float2 temp, wlen, w, u, v, next_w;

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

    // Algoritmo de la mariposa
    for (len = 2; len <= N1; len <<= 1)
    {
        angle = -2 * M_PI / len;
        wlen.x = cos(angle);
        wlen.y = sin(angle);
        for (i = 0; i < N1; i += len)
        {
            w.x = 1.0;
            w.y = 0.0;
            for (j = 0; j < len / 2; ++j)
            {
                u = x[fft_num * N1 + i + j];
                v.x = x[fft_num * N1 + i + j + len / 2].x * w.x - x[fft_num * N1 + i + j + len / 2].y * w.y;
                v.y = x[fft_num * N1 + i + j + len / 2].x * w.y + x[fft_num * N1 + i + j + len / 2].y * w.x;
                x[fft_num * N1 + i + j].x = u.x + v.x;
                x[fft_num * N1 + i + j].y = u.y + v.y;
                x[fft_num * N1 + i + j + len / 2].x = u.x - v.x;
                x[fft_num * N1 + i + j + len / 2].y = u.y - v.y;
                next_w.x = w.x * wlen.x - w.y * wlen.y;
                next_w.y = w.x * wlen.y + w.y * wlen.x;
                w = next_w;
            }
        }
    }
}
