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


qint64 Analysis::intRoundClosest(const qint64 n, const qint64 d)
{
    return ((n < 0) ^ (d < 0)) ? ((n - d/2)/d) : ((n + d/2)/d);
}
