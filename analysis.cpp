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

Eigen::MatrixXd Analysis::calcSavGolCoefs(int winSize, int polyOrder)
{
    Eigen::MatrixXd xm(polyOrder+1,winSize);
    Eigen::MatrixXd bm(polyOrder+1,polyOrder+1);

    for(int i=0; i<xm.rows(); i++)
    {
        for(int j=0; j<(int)xm.cols(); j++)
        {
            int z = j - (winSize/2);
            double val = pow((double)z,(double)i);
            xm(i,j) = val;
        }
    }

    for(int i=0; i<bm.rows(); i++)
    {
        for(int j=0; j<bm.cols(); j++)
            bm(i,j) = (i == j ? 1.0 : 0.0);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd,Eigen::FullPivHouseholderQRPreconditioner> svd(xm, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.solve(bm);
}

QVector<double> Analysis::savGolSmooth(const Eigen::MatrixXd coefs, int derivativeOrder, QVector<double> d, double dx)
{
    QVector<double> out(d.size());
    if(derivativeOrder >= coefs.cols())
        return out;

    Eigen::VectorXd c = coefs.col(derivativeOrder);
    int halfWin = c.rows()/2;
    double pf = static_cast<double>(factorial(derivativeOrder))/pow(dx,static_cast<double>(derivativeOrder));
    for(int i=halfWin; i<d.size()-halfWin; i++)
    {
        //apply savitsky-golay smoothing
        double val = 0.0;
        for(int j=0; j<c.rows(); j++)
            val += c(j)*d.at(i+j-halfWin);
        out[i] = pf*val;
    }

    return out;

}

int Analysis::factorial(int x)
{
    if(x>10)
        return 0;

    return (x == 1 || x == 0) ? 1 : x*factorial(x-1);
}

double Analysis::savGolSmoothPoint(int i, const Eigen::MatrixXd coefs, int derivativeOrder, QVector<double> d, double dx)
{
    if(derivativeOrder >= coefs.cols())
        return 0.0;

    if(i < 0 || i >= d.size())
        return 0.0;

    Eigen::VectorXd c = coefs.col(derivativeOrder);
    int halfWin = c.rows()/2;

    if(i < halfWin || i >= d.size()-halfWin)
        return d.at(i);

    double out = 0.0;
    for(int j=0; j<c.rows(); j++)
        out += c(j)*d.at(i+j-halfWin);

    double pf = static_cast<double>(factorial(derivativeOrder))/pow(dx,static_cast<double>(derivativeOrder));
    return pf*out;

}
