#include "ftworker.h"
#include <gsl/gsl_const.h>
#include <gsl/gsl_sf.h>

FtWorker::FtWorker(QObject *parent) :
    QObject(parent), real(NULL), work(NULL), d_numPnts(0), d_start(0.0), d_end(0.0)
{
}

QPair<QVector<QPointF>, double> FtWorker::doFT(const Fid f)
{
    if(f.size() < 2)
        return QPair<QVector<QPointF>,double>(QVector<QPointF>(),0.0);

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
    QVector<QPointF> spectrum;
    spectrum.reserve((int)ceil((double)d_numPnts/2.0));
    double spacing = fid.spacing();
    double probe = fid.probeFreq();
    double sign = 1.0;
    if(fid.sideband() == Fid::LowerSideband)
        sign = -1.0;

    //do the FT. See GNU Scientific Library documentation for details
    gsl_fft_real_transform (fftData.data(), 1, d_numPnts, real, work);

    //convert fourier coefficients into magnitudes. the coefficients are stored in half-complex format
    //see http://www.gnu.org/software/gsl/manual/html_node/Mixed_002dradix-FFT-routines-for-real-data.html
    //first point is DC; block it!
    spectrum << QPointF(probe,0.0);
//    spectrum << QPointF(probe,sqrt(fftData.at(0)*fftData.at(0)));
    double max = 0.0;
    int i;
    for(i=1; i<d_numPnts-i; i++)
    {
        //calculate x value
        double x1 = probe + sign*(double)i/(double)d_numPnts/spacing*1.0e-6;

        //calculate real and imaginary coefficients
        double coef_real = fftData.at(2*i-1);
        double coef_imag = fftData.at(2*i);

        //calculate magnitude and update max
        //note: Normalize output, and convert to mV
        double coef_mag = sqrt(coef_real*coef_real + coef_imag*coef_imag)/(double)d_numPnts*1000.0;
        max = qMax(max,coef_mag);

        spectrum.append(QPointF(x1,coef_mag));
    }
    if(i==d_numPnts-i)
       spectrum.append(QPointF(probe + sign*(double)i/(double)d_numPnts/spacing*1.0e-6,
                          sqrt(fftData.at(d_numPnts-1)*fftData.at(d_numPnts-1))/(double)d_numPnts*1000.0));

    //the signal is used for asynchronous purposes (in UI classes), and the return value for synchronous (in non-UI classes)
    emit ftDone(spectrum,max);
    return QPair<QVector<QPointF>,double>(spectrum,max);
}

Fid FtWorker::filterFid(const Fid fid)
{
    QVector<double> data = fid.toVector();

    //make a vector of points for display purposes
    QVector<QPointF> displayFid;
    displayFid.reserve(data.size());
    for(int i=0; i<data.size(); i++)
        displayFid.append(QPointF((double)i*fid.spacing()*1.0e6,data.at(i)));

    emit fidDone(displayFid);

    if(d_end < 0.01)
        d_end = fid.spacing()*fid.size()*1e6;

    qint64 start = 0, end = fid.size();

    if(d_start>0.0 && d_start < d_end)
    {
        start = (qint64)round(d_start/1e6/fid.spacing());
        start = qMax((qint64)0,start);
    }

    if(d_end > d_start && d_end < fid.spacing()*fid.size()*1e6)
    {
        end = (qint64)round(d_end/1e6/fid.spacing());
        end = qMin((qint64)fid.size(),end);
    }

    if(start + 1000 > end && start + 1000 <= (qint64)fid.size())
        end = start + 1000;
    else if(start + 1000 > end && start - 1000 >= 0)
        start -= 1000;
    else if(start + 1000 > end)
    {
        start = 0;
        end = fid.size();
    }

    for(int i=0;i<start && i <data.size(); i++)
        data.replace(i,0.0);
    for(int i=end; i<data.size(); i++)
        data.replace(i,0.0);

    int chunkSize = 50000;
    int choppedStart = ((int)start/chunkSize)*chunkSize;
    int choppedLength = qMin((((int)end-choppedStart+chunkSize)/chunkSize)*chunkSize,data.size()-choppedStart);

    //for synchronous use (eg the doFT function), return an FID object
    return Fid(fid.spacing(),fid.probeFreq(),data.mid(choppedStart,choppedLength));
}
