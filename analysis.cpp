#include "analysis.h"

#include <QtAlgorithms>
#include <math.h>

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


QList<int> Analysis::parseIntRanges(const QString str, int max)
{
    QList<int> out;

    QList<QString> l = str.split(QChar(','),QString::SkipEmptyParts);
    for(int i=0; i<l.size(); i++)
    {
        QString s = l.at(i).trimmed();
        if(s.contains(QChar('-')))
        {
            QList<QString> l2 = s.split(QChar('-'),QString::SkipEmptyParts);
            if(l2.size() != 2)
                continue;

            bool ok = false;
            int start = l2.at(0).trimmed().toInt(&ok);
            if(!ok || start < 1 || start > max)
                continue;
            int end = l2.at(1).trimmed().toInt(&ok);
            if(!ok || end < 1 || end > max)
                continue;

            if(end < start)
                std::swap(start,end);

            for(int j=start; j<=qMin(max,end); j++)
                out.append(j);
        }
        else
        {
            bool ok = false;
            int num = l.at(i).trimmed().toInt(&ok);
            if(!ok || num < 1 || num > max)
                continue;

            out.append(num);
        }
    }

    return out;
}


void Analysis::split(double dat[], int n, double val, int &i, int &j)
{
    if(i < 0 || i >= n || j<0 || j >= n)
        return;

    do
    {
        while(dat[i] < val)
            i++;

        while(val < dat[j])
            j--;

        if(i <= j)
        {
            double t = dat[i];
            dat[i] = dat[j];
            dat[j] = t;
            i++;
            j--;
        }

    } while(i <= j);
}

QPair<double,double> Analysis::medianFilterMeanStDev(double dat[], int n)
{
    int L = 0;
    int R = n - 1;
    int k = n/2 - 1 + (n%2);
    int size = n;
    double val;
    int i, j;
    double median;
    int totalReject = 0;

    while(true)
    {
        while(L < R)
        {
            val = dat[k];
            i = L;
            j = R;
            split(dat,n,val,i,j);
            if(j < k)
                L = i;
            if(k < i)
                R = j;
        }

        median = dat[k];
        int reject = 0;
        for(int z=k+1; z<size; z++)
        {
            if(dat[z] > 10.0*median)
                reject++;
        }

        if(reject == 0)
        {
            double mean = dat[0];
            double sumSq = 0.0;
            for(int z=1; z<size; z++)
            {
                double delta = dat[z]-mean;
                mean += delta/((double)z + 1.0);
                sumSq += delta*(dat[z]-mean);
            }
            //increase mean and stDev artificially if many points have been removed
            double scf = 1.0 + 2.0*(double)totalReject/(double)n;
            return qMakePair(scf*mean, scf*sqrt(sumSq/((double)size - 1.0)));
        }

        size -= reject;
        totalReject += reject;
        L = 0;
        R = size - 1;
        k -= (reject/2);
    }

    //not reached
    return qMakePair(0.0,1.0);
}
