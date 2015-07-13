#include "analysis.h"

quint32 Analysis::nextPowerOf2(quint32 n)
{
    if(n == 0)
        return 1;

    n--;
    for(int i = 1; i<32; i*=2)
        n |= n >> i;

    return n+1;
}
