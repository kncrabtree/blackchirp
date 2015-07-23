#include "ftworker.h"

#include <QTime>

#include <gsl/gsl_const.h>
#include <gsl/gsl_sf.h>

#include "analysis.h"

FtWorker::FtWorker(QObject *parent) :
    QObject(parent), real(NULL), work(NULL), d_numPnts(0), d_start(0.0), d_end(0.0), d_pzf(0)
{
}

QPair<QVector<QPointF>, double> FtWorker::doFT(const Fid f)
{
    if(f.size() < 2)
        return QPair<QVector<QPointF>,double>(QVector<QPointF>(),0.0);

    double rawSize = static_cast<double>(f.size());

    //first, apply any filtering that needs to be done
    Fid fid = filterFid(f);

    //might need to allocate or reallocate workspace and wavetable
    if(fid.size() != d_numPnts)
    {
        d_numPnts = fid.size();

        //free memory if this is a reallocation
        if(real)
        {
            gsl_fft_real_wavetable_free(real);
            gsl_fft_real_workspace_free(work);
        }

        real = gsl_fft_real_wavetable_alloc(d_numPnts);
        work = gsl_fft_real_workspace_alloc(d_numPnts);
    }

    //prepare storage
    QVector<double> fftData(fid.toVector());
    int spectrumSize = d_numPnts/2 + 1;
    QVector<QPointF> spectrum(spectrumSize);
    double spacing = fid.spacing()*1.0e6;
    double probe = fid.probeFreq();
    double sign = 1.0;
    if(fid.sideband() == BlackChirp::LowerSideband)
        sign = -1.0;


    //do the FT. See GNU Scientific Library documentation for details
    gsl_fft_real_transform (fftData.data(), 1, d_numPnts, real, work);


    //convert fourier coefficients into magnitudes. the coefficients are stored in half-complex format
    //see http://www.gnu.org/software/gsl/manual/html_node/Mixed_002dradix-FFT-routines-for-real-data.html
    //first point is DC; block it!
    if(fid.sideband() == BlackChirp::UpperSideband)
        spectrum[0] = QPointF(probe,0.0);
    else
        spectrum[spectrumSize-1] = QPointF(probe,0.0);

    double max = 0.0;
    int i;
    double np = static_cast<double>(d_numPnts);
    for(i=1; i<d_numPnts-i; i++)
    {
        //calculate x value
        double x1 = probe + sign*(double)i/np/spacing;

        //calculate real and imaginary coefficients
        double coef_real = fftData.at(2*i-1);
        double coef_imag = fftData.at(2*i);

        //calculate magnitude and update max
        //note: Normalize output, and convert to mV
        double coef_mag = sqrt(coef_real*coef_real + coef_imag*coef_imag)/rawSize;
        max = qMax(max,coef_mag);

        if(fid.sideband() == BlackChirp::UpperSideband)
            spectrum[i] = QPointF(x1,coef_mag);
        else
            spectrum[spectrumSize-1-i] = QPointF(x1,coef_mag);
    }
    if(i==d_numPnts-i)
    {
        if(fid.sideband() == BlackChirp::UpperSideband)
            spectrum[i] = QPointF(probe + sign*(double)i/np/spacing,
                          sqrt(fftData.at(d_numPnts-1)*fftData.at(d_numPnts-1))/rawSize);
        else
            spectrum[spectrumSize-1-i] = QPointF(probe + sign*(double)i/np/spacing,
                          sqrt(fftData.at(d_numPnts-1)*fftData.at(d_numPnts-1))/rawSize);
    }

    //the signal is used for asynchronous purposes (in UI classes), and the return value for synchronous (in non-UI classes)
    emit ftDone(spectrum,max);

    return QPair<QVector<QPointF>,double>(spectrum,max);
}

void FtWorker::doFtDiff(const Fid ref, const Fid diff)
{
    if(ref.size() != diff.size() || ref.sideband() != diff.sideband())
        return;

    if(!qFuzzyCompare(ref.spacing(),diff.spacing()) || !qFuzzyCompare(ref.probeFreq(),diff.probeFreq()))
        return;

    blockSignals(true);
    auto r = doFT(ref);
    auto d = doFT(diff);
    blockSignals(false);

    double max = r.first.first().y() - d.first.first().y();
    double min = max;


    for(int i=0; i<r.first.size() && i<d.first.size(); i++)
    {
        r.first[i].setY(r.first.at(i).y() - d.first.at(i).y());
        min = qMin(r.first.at(i).y(),min);
        max = qMax(r.first.at(i).y(),max);
    }

    emit ftDiffDone(r.first,min,max);

}

Fid FtWorker::filterFid(const Fid fid)
{
//    QVector<double> data = fid.toVector();

//    //make a vector of points for display purposes
//    QVector<QPointF> displayFid;
//    displayFid.reserve(data.size());
//    for(int i=0; i<data.size(); i++)
//        displayFid.append(QPointF((double)i*fid.spacing()*1.0e6,data.at(i)));

//    emit fidDone(fid.toXY());

    Fid out = fid;
    QVector<qint64> data = out.rawData();
    if(d_fidData.size() < data.size())
        d_fidData.resize(data.size());
    d_fidData.fill(0);

    bool fStart = (d_start > 0.001);
    bool fEnd = (d_end > 0.001);

    for(int i=0; i<data.size(); i++)
    {
        if(fStart && static_cast<double>(i)*fid.spacing() < d_start*1e-6)
            continue;
        else if(fEnd && static_cast<double>(i)*fid.spacing() > d_end*1e-6)
            continue;
        else
            d_fidData[i] = data.at(i);
    }

    if(d_pzf > 0 && d_pzf <= 4)
    {
        int filledSize = Analysis::nextPowerOf2(2*data.size()) << (d_pzf-1);
        if(d_fidData.size() != filledSize)
            d_fidData.resize(filledSize);
    }
    else if(d_pzf == 0)
    {
        if(d_fidData.size() != data.size())
            d_fidData.resize(data.size());
    }

    out.setData(d_fidData);

    return out;

}
