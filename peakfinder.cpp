#include "peakfinder.h"

#include <eigen3/Eigen/SVD>

#include "analysis.h"

PeakFinder::PeakFinder(QObject *parent) : QObject(parent)
{
    calcCoefs(11,6);
}

QList<QPointF> PeakFinder::findPeaks(const QVector<QPointF> ft, double minF, double maxF, double minSNR)
{
    if(ft.size() < d_window)
    {
        emit peakList(QList<QPointF>());
        return QList<QPointF>();
    }

    //calculate smoothed second derivative
    QVector<double> smth(ft.size());
    QVector<double> yDat(ft.size());
    Eigen::VectorXd c = d_coefs.col(2);
    int halfWin = c.rows()/2;

    int startIndex = qAbs((minF - ft.first().x())/(ft.last().x()-ft.first().x())*((double)ft.size()-1.0));
    int endIndex = qAbs((maxF - ft.first().x())/(ft.last().x()-ft.first().x())*((double)ft.size()-1.0));
    if(endIndex < startIndex)
        qSwap(startIndex,endIndex);

    startIndex = qMax(halfWin,startIndex);
    endIndex = qMin(ft.size()-halfWin,endIndex);

    for(int i=startIndex; i<endIndex; i++)
    {
        //apply savitsky-golay smoothing, ignoring prefactor of 2/h^2 because we're only interested in local minima
        double val = 0.0;
        for(int j=0; j<c.rows(); j++)
            val += c(j)*ft.at(i+j-halfWin).y();
        smth[i] = val;
        yDat[i] = ft.at(i).y();
    }

    //build a noise model
    int chunks = 100;
    int chunkSize = (endIndex-startIndex)/chunks;
    QList<QPair<double,double>> blParams;
    for(int i=0; i<chunks; i++)
    {
        QVector<double> dat;
        if(i+1 == chunks)
            dat = yDat.mid(startIndex + i*chunkSize, endIndex - (startIndex + i*chunkSize));
        else
            dat = yDat.mid(startIndex + i*chunkSize, chunkSize);

        //median filter iteratively calculates median and removes any points that are 10 times greater than the median
        //Once all large points are rejected, it calculates and returns the mean and standard deviation
        blParams.append(Analysis::medianFilterMeanStDev(dat.data(),dat.size()));
    }

    QList<QPointF> out;
    for(int i = startIndex+2; i<endIndex-2; i++)
    {
        int thisChunk = (i-startIndex)/chunkSize;
        double mean = blParams.at(thisChunk).first;
        double stDev = blParams.at(thisChunk).second;
        double thisSNR = (yDat.at(i)-mean)/stDev;
        if(thisSNR >= minSNR)
        {
            //intensity is high enough; ID a peak by a minimum in 2nd derivative
            if(((smth.at(i-2) > smth.at(i-1)) && (smth.at(i-1) > smth.at(i)) && (smth.at(i) < smth.at(i+1))) ||
                    ((smth.at(i-1) > smth.at(i)) && (smth.at(i) < smth.at(i+1)) && (smth.at(i+1) < smth.at(i+2))) )
                out.append(ft.at(i));
        }
    }

    emit peakList(out);
    return out;

}

void PeakFinder::calcCoefs(int winSize, int polyOrder)
{
    if(polyOrder < 2)
        return;

    if(!(winSize % 2))
        return;

    if(winSize < polyOrder + 1)
        return;

    d_window = winSize;
    d_polyOrder = polyOrder;

    Eigen::MatrixXd xm(d_polyOrder+1,d_window);
    Eigen::MatrixXd bm(d_polyOrder+1,d_polyOrder+1);

    for(int i=0; i<xm.rows(); i++)
    {
        for(int j=0; j<(int)xm.cols(); j++)
        {
            int z = j - (d_window/2);
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
    d_coefs = svd.solve(bm);

}

