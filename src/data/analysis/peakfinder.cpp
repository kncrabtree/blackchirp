#include <data/analysis/peakfinder.h>

#include <data/analysis/analysis.h>

PeakFinder::PeakFinder(QObject *parent) : QObject(parent)
{
    calcCoefs(11,6);
}

QVector<QPointF> PeakFinder::findPeaks(const Ft ft, double minF, double maxF, double minSNR)
{
    if(ft.size() < d_window)
    {
        emit peakList({});
        return {};
    }

    //calculate smoothed second derivative
    QVector<double> smth(ft.size());
    QVector<double> yDat(ft.size());
    Eigen::VectorXd c = d_coefs.col(2);
    int halfWin = c.rows()/2;

    int startIndex = qAbs((minF - ft.constFirst().x())/(ft.constLast().x()-ft.constFirst().x())*((double)ft.size()-1.0));
    int endIndex = qAbs((maxF - ft.constFirst().x())/(ft.constLast().x()-ft.constFirst().x())*((double)ft.size()-1.0));
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
    int chunkSize = (endIndex-startIndex + 1)/chunks + 1;
    QVector<QPair<double,double>> blParams;
    for(int i=0; i<chunks; i++)
    {
        QVector<double> dat;
        dat = yDat.mid(startIndex + i*chunkSize, chunkSize);

        //median filter iteratively calculates median and removes any points that are 10 times greater than the median
        //Once all large points are rejected, it calculates and returns the mean and standard deviation
        blParams.append(Analysis::medianFilterMeanStDev(dat.data(),dat.size()));
    }

    QVector<QPointF> out;
    for(int i = startIndex+2; i<endIndex-2; i++)
    {
        int thisChunk = qBound(0,(i-startIndex)/chunkSize,chunks-1);
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

    d_coefs = Analysis::calcSavGolCoefs(d_window,d_polyOrder);

}

