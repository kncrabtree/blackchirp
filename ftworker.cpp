#include "ftworker.h"

#include <QTime>

#include <gsl/gsl_const.h>
#include <gsl/gsl_sf.h>

FtWorker::FtWorker(QObject *parent) :
    QObject(parent), real(NULL), work(NULL), d_numPnts(0), d_start(0.0), d_end(0.0), d_pzf(0), d_removeDC(false), d_showProcessed(false), d_scaling(1.0), d_ignoreZone(50.0), d_recalculateWinf(true)
{
}

QPair<QVector<QPointF>, double> FtWorker::doFT(const Fid fid)
{
    if(fid.size() < 2)
        return QPair<QVector<QPointF>,double>(QVector<QPointF>(),0.0);

    double rawSize = static_cast<double>(fid.size());

    //first, apply any filtering that needs to be done
    QVector<double> fftData = filterFid(fid);

    if(d_showProcessed)
        prepareForDisplay(fftData,fid.spacing());

    //might need to allocate or reallocate workspace and wavetable
    if(fftData.size() != d_numPnts)
    {
        d_numPnts = fftData.size();

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
        double coef_mag = sqrt(coef_real*coef_real + coef_imag*coef_imag)/rawSize*d_scaling;

        //only update max if we're 50 MHz away from LO
        if(qAbs(probe-x1) > d_ignoreZone)
            max = qMax(max,coef_mag);

        if(fid.sideband() == BlackChirp::UpperSideband)
            spectrum[i] = QPointF(x1,coef_mag);
        else
            spectrum[spectrumSize-1-i] = QPointF(x1,coef_mag);
    }
    if(i==d_numPnts-i)
    {
        QPointF p(probe + sign*(double)i/np/spacing,
                   sqrt(fftData.at(d_numPnts-1)*fftData.at(d_numPnts-1))/rawSize*d_scaling);

        if(fid.sideband() == BlackChirp::UpperSideband)
            spectrum[i] = p;
        else
            spectrum[spectrumSize-1-i] = p;

        //only update max if we're 50 MHz away from LO
        if(qAbs(probe-p.x()) > d_ignoreZone)
            max = qMax(max,p.y());

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

    if(d_showProcessed)
        prepareForDisplay(diff);

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

QVector<double> FtWorker::filterFid(const Fid fid)
{

    QVector<double> out(fid.size());
    QVector<double> data = fid.toVector();

    bool fStart = (d_start > 0.001);
    bool fEnd = (d_end > 0.001);

    int si = qBound(0, static_cast<int>(floor(d_start*1e-6/fid.spacing())), fid.size()-1);
    int ei = qBound(0,static_cast<int>(ceil(d_end*1e-6/fid.spacing())), fid.size()-1);
    if(!fStart || si - ei >= 0)
        si = 0;
    if(!fEnd || ei <= si)
        ei = fid.size()-1;

    int n = ei - si + 1;
    makeWinf(n,d_currentWinf);

    if(d_removeDC)
    {
        //calculate average of samples in the FT range, then subtract that from each point
        //use Kahan summation
        double sum = 0.0;
        double c = 0.0;
        for(int i=0; i<data.size(); i++)
        {
            if(i < si)
                continue;

            if(i > ei)
                break;

            float y = data.at(y) - c;
            float t = sum + y;
            c = (t-sum) - y;
            sum = t;
        }

        double avg = sum/static_cast<double>(n);
        for(int i=0; i<data.size(); i++)
        {
            if(i < si)
                continue;

            if(i > ei)
                break;

            data[i] -= avg;
        }

    }

    for(int i=0; i<data.size(); i++)
    {
        if(i < si)
            continue;

        if(i > ei)
            break;

        if(d_currentWinf == BlackChirp::Boxcar)
            out[i] = data.at(i);
        else
            out[i] = data.at(i)*d_winf.at(i-si);
    }

    if(d_pzf > 0 && d_pzf <= 4)
    {
        int filledSize = Analysis::nextPowerOf2(2*data.size()) << (d_pzf-1);
        if(out.size() != filledSize)
            out.resize(filledSize);
    }

    return out;

}

void FtWorker::prepareForDisplay(const QVector<double> fid, double spacing)
{
    QVector<QPointF> out(fid.size());
    for(int i=0; i<out.size(); i++)
    {
        out[i].setX(spacing*static_cast<double>(i));
        out[i].setY(fid.at(i));
    }

    emit fidDone(out);
}

void FtWorker::prepareForDisplay(const Fid fid)
{
    return prepareForDisplay(filterFid(fid),fid.spacing());
}

void FtWorker::makeWinf(int n, BlackChirp::FtWindowFunction f)
{
    if(f == d_currentWinf && d_winf.size() == n && !d_recalculateWinf)
        return;

    if(d_winf.size() != n)
        d_winf.resize(n);

    switch(f)
    {
    case BlackChirp::Bartlett:
        winBartlett(n);
        break;
    case BlackChirp::Blackman:
        winBlackman(n);
        break;
    case BlackChirp::BlackmanHarris:
        winBlackmanHarris(n);
        break;
    case BlackChirp::Hamming:
        winHamming(n);
        break;
    case BlackChirp::Hanning:
        winHanning(n);
        break;
    case BlackChirp::KaiserBessel14:
        winKaiserBessel(n,14.0);
        break;
    case BlackChirp::Boxcar:
    default:
        d_winf.fill(1.0);
        break;
    }

    d_recalculateWinf = false;
}

void FtWorker::winBartlett(int n)
{
    double a = (static_cast<double>(n)-1.0)/2.0;
    double b = 2.0/(static_cast<double>(n)-1.0);
    for(int i=0; i<n; i++)
        d_winf[i] = b*(a-qAbs(static_cast<double>(i)-a));
}

void FtWorker::winBlackman(int n)
{
    double N = static_cast<double>(n);
    double p2n = 2.0*M_PI/N;
    double p4n = 4.0*M_PI/N;
    for(int i=0; i<n; i++)
    {
        double I = static_cast<double>(i);
        d_winf[i] = 0.42 - 0.5*cos(p2n*I) + 0.08*cos(p4n*I);
    }

}

void FtWorker::winBlackmanHarris(int n)
{
    double N = static_cast<double>(n);
    double p2n = 2.0*M_PI/N;
    double p4n = 4.0*M_PI/N;
    double p6n = 6.0*M_PI/N;
    for(int i=0; i<n; i++)
    {
        double I = static_cast<double>(i);
        d_winf[i] = 0.35875 - 0.48829*cos(p2n*I) + 0.14128*cos(p4n*I) - 0.01168*cos(p6n*I);
    }
}

void FtWorker::winHamming(int n)
{
    double N = static_cast<double>(n)-1.0;
    double p2n = 2.0*M_PI/N;
    for(int i=0; i<n; i++)
    {
        double I = static_cast<double>(i);
        d_winf[i] = 0.54 - 0.46*cos(p2n*I);
    }
}

void FtWorker::winHanning(int n)
{
    double N = static_cast<double>(n)-1.0;
    double p2n = 2.0*M_PI/N;
    for(int i=0; i<n; i++)
    {
        double I = static_cast<double>(i);
        d_winf[i] = 0.5 - 0.5*cos(p2n*I);
    }
}

void FtWorker::winKaiserBessel(int n, double beta)
{
    double Ibeta = gsl_sf_bessel_I0(beta);
    double n2 = 2.0/(static_cast<double>(n)-1.0);
    for(int i=0; i<n; i++)
    {
        double I = static_cast<double>(i);
        double arg = beta*sqrt(1.0-(n2*I-1.0)*(n2*I-1.0));
        double bsl = gsl_sf_bessel_I0(arg);
        if(gsl_isinf(bsl) || gsl_isnan(bsl))
            d_winf[i] = 0.0;
        else
            d_winf[i] = bsl/Ibeta;
    }
}
