#include "ftworker.h"

#include <QTime>

#include <gsl/gsl_const.h>
#include <gsl/gsl_sf.h>

FtWorker::FtWorker(int i, QObject *parent) :
    QObject(parent), d_id(i), real(NULL), work(NULL), d_numPnts(0), p_spline(nullptr),
    p_accel(nullptr), d_numSplinePoints(0)
{
    d_lastProcSettings = FidProcessingSettings { -1.0, -1.0, 0, false, BlackChirp::FtPlotuV, 50.0, BlackChirp::Boxcar };
}

FtWorker::~FtWorker()
{
    if(p_spline != nullptr)
        gsl_spline_free(p_spline);
    if(p_accel != nullptr)
        gsl_interp_accel_free(p_accel);
    if(real != nullptr)
    {
        gsl_fft_real_wavetable_free(real);
        gsl_fft_real_workspace_free(work);
    }
}

Ft FtWorker::doFT(const Fid fid, const FidProcessingSettings &settings)
{
    if(fid.size() < 2)
    {
        emit fidDone(QVector<QPointF>(),d_id);
        emit ftDone(Ft(), d_id);
        return Ft();
    }

    double rawSize = static_cast<double>(fid.size());

    //first, apply any filtering that needs to be done
    QVector<double> fftData = filterFid(fid,settings);
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
    Ft spectrum(spectrumSize,fid.probeFreq());

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
    //always make sure that data go from low to high frequency
    if(fid.sideband() == BlackChirp::UpperSideband)
        spectrum.setPoint(0,QPointF(probe,0.0),settings.autoScaleIgnoreMHz);
    else
        spectrum.setPoint(spectrumSize-1,QPointF(probe,0.0),settings.autoScaleIgnoreMHz);

    int i;
    double np = static_cast<double>(d_numPnts);
    double scf = BlackChirp::getFtScalingFactor(settings.units);
    for(i=1; i<d_numPnts-i; i++)
    {
        //calculate x value
        double x1 = probe + sign*(double)i/np/spacing;

        //calculate real and imaginary coefficients
        double coef_real = fftData.at(2*i-1);
        double coef_imag = fftData.at(2*i);

        //calculate magnitude and update max
        //note: Normalize output, and convert to mV
        double coef_mag = sqrt(coef_real*coef_real + coef_imag*coef_imag)/rawSize*scf;

        if(fid.sideband() == BlackChirp::UpperSideband)
            spectrum.setPoint(i, QPointF(x1,coef_mag),settings.autoScaleIgnoreMHz);
        else
            spectrum.setPoint(spectrumSize-1-i,QPointF(x1,coef_mag),settings.autoScaleIgnoreMHz);
    }
    if(i==d_numPnts-i)
    {
        QPointF p(probe + sign*(double)i/np/spacing,
                   sqrt(fftData.at(d_numPnts-1)*fftData.at(d_numPnts-1))/rawSize*scf);

        if(fid.sideband() == BlackChirp::UpperSideband)
            spectrum.setPoint(i,p,settings.autoScaleIgnoreMHz);
        else
            spectrum.setPoint(spectrumSize-1-i,p,settings.autoScaleIgnoreMHz);

        //only update max if we're 50 MHz away from LO
//        if(qAbs(probe-p.x()) > d_ignoreZone)
//            max = qMax(max,p.y());

    }

    //the signal is used for asynchronous purposes (in UI classes), and the return value for synchronous (in non-UI classes)
    emit ftDone(spectrum,d_id);

    d_lastProcSettings = settings;

    return spectrum;
}

void FtWorker::doFtDiff(const Fid ref, const Fid diff, const FidProcessingSettings &settings)
{
    if(ref.size() != diff.size() || ref.sideband() != diff.sideband())
        return;

    if(!qFuzzyCompare(ref.spacing(),diff.spacing()))
        return;

    blockSignals(true);
    Ft r = doFT(ref,settings);
    Ft d = doFT(diff,settings);
    blockSignals(false);

    Ft out(r.size(),r.loFreq());

    if(qFuzzyCompare(r.loFreq(),d.loFreq()))
    {

        for(int i=0; i<r.size() && i<d.size(); i++)
        {
            auto p = r.at(i);
            p.setY(p.y() - d.at(i).y());
            out.setPoint(i,p);
        }
    }
    else
    {
        Ft drs = resample(r.loFreq(),r.xSpacing(),d);
        out = Ft(r.size() + drs.size(),r.loFreq());
        int rIndex = 0, dIndex = 0, totalPoints = 0;
        bool done = false;
        while(!done)
        {
            if(rIndex < r.size() && dIndex < drs.size())
            {
                double rx = r.at(rIndex).x();
                double dx = drs.at(dIndex).x();

                if(qAbs(rx-dx) < r.xSpacing()) //frequencies the same; difference and increment both
                {
                    out.setPoint(totalPoints,QPointF(rx,r.at(rIndex).y()-drs.at(dIndex).y()));
                    dIndex++;
                    rIndex++;
                }
                else
                {
                    if(rx < dx)
                    {
                        if(rIndex < r.size())
                        {
                            out.setPoint(totalPoints,QPointF(rx,r.at(rIndex).y()));
                            rIndex++;
                        }
                        else
                        {
                            out.setPoint(totalPoints,QPointF(dx,-drs.at(dIndex).y()));
                            dIndex++;
                        }
                    }
                    else
                    {
                        if(dIndex < drs.size())
                        {
                            out.setPoint(totalPoints,QPointF(dx,-drs.at(dIndex).y()));
                            dIndex++;
                        }
                        else
                        {
                            out.setPoint(totalPoints,QPointF(rx,r.at(rIndex).y()));
                            rIndex++;
                        }
                    }
                }
            }
            else
            {
                if(rIndex < r.size())
                {
                    out.setPoint(totalPoints,QPointF(r.at(rIndex).x(),r.at(rIndex).y()));
                    rIndex++;
                }
                else
                {
                    out.setPoint(totalPoints,QPointF(drs.at(dIndex).x(),-drs.at(dIndex).y()));
                    dIndex++;
                }
            }

            totalPoints++;

            if(rIndex == r.size() && dIndex == drs.size())
                done = true;
        }

        out.resize(totalPoints);

    }

    d_lastProcSettings = settings;

    emit ftDiffDone(out);

}

Ft FtWorker::processSideband(const FidList fl, const FtWorker::FidProcessingSettings &settings, BlackChirp::Sideband sb)
{
    //this will FT all of the FIDs and resample them on a common grid
    QList<Ft> ftList = makeSidebandList(fl,settings,sb);

    if(ftList.size() == 1)
    {
        emit ftDone(ftList.first(),d_id);
        return ftList.first();
    }

    QVector<int> indices;
    indices.resize(ftList.size());

    QVector<QPointF> dataPointsList;
    dataPointsList.reserve(ftList.size()*ftList.first().size());

    //want to make sure frequency increases monotonically as we iterate through fidlist
    //Fts ALWAYS go from low frequency to high
    if(ftList.first().loFreq() > ftList.last().loFreq())
        std::reverse(ftList.begin(),ftList.end());

    while(true)
    {
        double thisPointY = 0.0;
        double thisPointX = 0.0;
        double numPoints = 0.0;

        for(int i=0; i<indices.size(); i++)
        {
            if(indices.at(i) < ftList.at(i).size())
            {
                thisPointX = ftList.at(i).at(indices.at(i)).x();
                break;
            }
        }

        for(int i=0; i<indices.size(); i++)
        {
            if(indices.at(i) < ftList.at(i).size())
            {
                if(qAbs(thisPointX - ftList.at(i).at(indices.at(i)).x()) < ftList.at(i).xSpacing())
                {
                    double y = ftList.at(i).at(indices.at(i)).y();
                    if(y>0)
                    {
                        thisPointY += log10(y);
                        numPoints += 1.0;
                    }
                    indices[i]++;
                }
            }
        }

        if(numPoints < 1.0)
            dataPointsList.append(QPointF(thisPointX,thisPointY));
        else
            dataPointsList.append(QPointF(thisPointX,pow(10.0,thisPointY/numPoints)));

        bool done = true;
        for(int i=0; i<indices.size(); i++)
        {
            if(indices.at(i) < ftList.at(i).size())
            {
                done = false;
                break;
            }
        }

        if(done)
            break;
    }

    Ft out(dataPointsList.size(),0.0);
    for(int i=0; i<dataPointsList.size(); i++)
        out.setPoint(i,dataPointsList.at(i));

    emit ftDone(out,d_id);
    return out;
}

QList<Ft> FtWorker::makeSidebandList(const FidList fl, const FidProcessingSettings &settings, BlackChirp::Sideband sb)
{
    if(fl.isEmpty())
        return QList<Ft>();

    Fid f = fl.first();
    f.setSideband(sb);

    QList<Ft> out;

    blockSignals(true);
    Ft ft1 = doFT(f,settings);
    out << ft1;

    double f0 = ft1.loFreq();
    double sp = ft1.xSpacing();
    for(int i=1; i<fl.size(); i++)
    {
        f = fl.at(i);
        f.setSideband(sb);
        ft1 = doFT(f,settings);
        auto rsft = resample(f0,sp,ft1);
        out << rsft;
    }
    blockSignals(false);

    return out;

}

QVector<double> FtWorker::filterFid(const Fid fid, const FidProcessingSettings &settings)
{

    QVector<double> out(fid.size());
    QVector<double> data = fid.toVector();

    bool fStart = (settings.startUs > 0.001);
    bool fEnd = (settings.endUs > 0.001);

    int si = qBound(0, static_cast<int>(floor(settings.startUs*1e-6/fid.spacing())), fid.size()-1);
    int ei = qBound(0,static_cast<int>(ceil(settings.endUs*1e-6/fid.spacing())), fid.size()-1);
    if(!fStart || si - ei >= 0)
        si = 0;
    if(!fEnd || ei <= si)
        ei = fid.size()-1;

    int n = ei - si + 1;
    makeWinf(n,settings.windowFunction);

    if(settings.removeDC)
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

            float y = data.at(i) - c;
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

        if(settings.windowFunction == BlackChirp::Boxcar)
            out[i] = data.at(i);
        else
            out[i] = data.at(i)*d_winf.at(i-si);
    }

    if(settings.zeroPadFactor > 0 && settings.zeroPadFactor <= 4)
    {
        int filledSize = Analysis::nextPowerOf2(2*data.size()) << (settings.zeroPadFactor-1);
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

    emit fidDone(out,d_id);
}

Ft FtWorker::resample(double f0, double spacing, const Ft ft)
{
    if(ft.isEmpty() || ft.size() < 2 || spacing == 0.0)
        return Ft();

    spacing = qAbs(spacing);
    double thisSpacing = ft.at(1).x() - ft.at(0).x();

    if(qFuzzyCompare(f0,ft.loFreq()) && qFuzzyCompare(qAbs(spacing),qAbs(thisSpacing)))
        return Ft();


    double minF = ft.minFreq();
    double maxF = ft.maxFreq();
    double direction = thisSpacing > 0 ? 1.0 : -1.0;

    //find sample point closest to, but greater than minf
    double firstPt = f0 + ceil((minF-f0)/spacing)*spacing;

    int numPoints = floor((maxF-firstPt)/spacing);

    //allocate or reallocate gsl_spline object
    if(ft.size() != d_numSplinePoints)
    {
        d_numSplinePoints = ft.size();

        if(p_spline != nullptr)
            gsl_spline_free(p_spline);

        p_spline = gsl_spline_alloc(gsl_interp_cspline,d_numSplinePoints);

        if(p_accel != nullptr)
        {
            gsl_interp_accel_free(p_accel);
            p_accel = gsl_interp_accel_alloc();
        }
    }

    if(p_accel == nullptr)
        p_accel = gsl_interp_accel_alloc();

    //set up spline object with FT data
    auto xd = ft.xData();
    auto yd = ft.yData();
    bool reverse = false;
    if(xd.last() < xd.first())
    {
        reverse = true;
        std::reverse(xd.begin(),xd.end());
        std::reverse(yd.begin(),yd.end());
    }

    gsl_spline_init(p_spline,xd.constData(),yd.constData(),d_numSplinePoints);

    int index = 0;
    Ft out(numPoints,ft.loFreq());

    while(index < numPoints)
    {
        double x = firstPt + static_cast<double>(index)*spacing*direction;
        double y = gsl_spline_eval(p_spline,x,p_accel);
        int i = index;
        if(reverse)
            i = numPoints - index - 1;
        out.setPoint(i,QPointF(x,y));
        index++;
    }

    return out;

}

void FtWorker::makeWinf(int n, BlackChirp::FtWindowFunction f)
{
    if(f == d_lastProcSettings.windowFunction && d_winf.size() == n)
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
