#include <data/analysis/ftworker.h>

#include <QTime>
#include <QReadWriteLock>
#include <QReadLocker>
#include <QWriteLocker>

#include <gsl/gsl_const.h>
#include <gsl/gsl_sf.h>

FtWorker::FtWorker(QObject *parent) :
    QObject(parent), real(NULL), work(NULL), d_numPnts(0), p_spline(nullptr),
    p_accel(nullptr), d_numSplinePoints(0)
{    
    pu_fftLock = std::make_unique<QReadWriteLock>();
    pu_splineLock = std::make_unique<QReadWriteLock>();
    pu_winfLock = std::make_unique<QReadWriteLock>();
}

FtWorker::~FtWorker()
{
    QWriteLocker l(pu_splineLock.get());
    QWriteLocker l2(pu_fftLock.get());
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

Ft FtWorker::doFT(const Fid fid, const FidProcessingSettings &settings, int id, bool doubleSideband)
{
    if(fid.size() < 2)
    {
        emit fidDone({},1.0,0.0,0.0,id);
        emit ftDone(Ft(), id);
        return Ft();
    }

    double rawSize = static_cast<double>(fid.size());

    //first, apply any filtering that needs to be done
    auto fidResult = filterFid(fid,settings);
    emit fidDone(fidResult.fid,fid.spacing()*1e6,fidResult.min,fidResult.max,id);
    auto fftData = fidResult.fid;
    auto s = fftData.size();


    auto spacing = fid.spacing()*1.0e6;
    auto probe = fid.probeFreq();
    double ftSpacing = 1.0/rawSize/spacing;
    int spectrumSize = s/2 + 1;
    if(doubleSideband)
        spectrumSize = s;

    double bandwidth = (spectrumSize-1)*ftSpacing;
    Ft spectrum;
    if(doubleSideband)
        spectrum = Ft(spectrumSize,probe-bandwidth/2.0,ftSpacing,probe);
    else if(fid.sideband() == RfConfig::UpperSideband)
        spectrum = Ft(spectrumSize,probe,ftSpacing,probe);
    else
        spectrum = Ft(spectrumSize,probe-bandwidth,ftSpacing,probe);

    //might need to allocate or reallocate workspace and wavetable
    QWriteLocker l(pu_fftLock.get());
    if(s != d_numPnts)
    {
        //free memory if this is a reallocation
        //this should be very infrequent
        d_numPnts = s;
        if(real)
        {
            gsl_fft_real_wavetable_free(real);
            gsl_fft_real_workspace_free(work);
        }

        real = gsl_fft_real_wavetable_alloc(d_numPnts);
        work = gsl_fft_real_workspace_alloc(d_numPnts);
    }

    //do the FT. See GNU Scientific Library documentation for details
    gsl_fft_real_transform (fftData.data(), 1, d_numPnts, real, work);
    l.unlock();




    //convert fourier coefficients into magnitudes. the coefficients are stored in half-complex format
    //see http://www.gnu.org/software/gsl/manual/html_node/Mixed_002dradix-FFT-routines-for-real-data.html
    //first point is DC; block it!
    //always make sure that data go from low to high frequency
    if(doubleSideband)
        spectrum.setPoint(spectrumSize/2,0.0,settings.autoScaleIgnoreMHz);
    else if(fid.sideband() == RfConfig::UpperSideband)
        spectrum.setPoint(0,0.0,settings.autoScaleIgnoreMHz);
    else
        spectrum.setPoint(spectrumSize-1,0.0,settings.autoScaleIgnoreMHz);

    int i;
    double scf = pow(10.,static_cast<double>(settings.units))/rawSize;
    for(i=1; i<s-i; i++)
    {

        //calculate real and imaginary coefficients
        double coef_real = fftData.at(2*i-1);
        double coef_imag = fftData.at(2*i);

        //calculate magnitude and update max
        //note: Normalize output, and convert to mV
        double coef_mag = sqrt(coef_real*coef_real + coef_imag*coef_imag)*scf;

        if(doubleSideband)
        {
            spectrum.setPoint(spectrumSize/2+i,coef_mag,settings.autoScaleIgnoreMHz);
            spectrum.setPoint(spectrumSize/2-i,coef_mag,settings.autoScaleIgnoreMHz);
        }
        else if(fid.sideband() == RfConfig::UpperSideband)
            spectrum.setPoint(i,coef_mag,settings.autoScaleIgnoreMHz);
        else
            spectrum.setPoint(spectrumSize-1-i,coef_mag,settings.autoScaleIgnoreMHz);
    }
    if(i==s-i)
    {
        double d = sqrt(fftData.at(s-1)*fftData.at(s-1))*scf;

        if(doubleSideband)
        {
            spectrum.setPoint(spectrumSize/2+i,d,settings.autoScaleIgnoreMHz);
            spectrum.setPoint(spectrumSize/2-i,d,settings.autoScaleIgnoreMHz);
        }
        if(fid.sideband() == RfConfig::UpperSideband)
            spectrum.setPoint(i,d,settings.autoScaleIgnoreMHz);
        else
            spectrum.setPoint(spectrumSize-1-i,d,settings.autoScaleIgnoreMHz);
    }

    spectrum.setNumShots(fid.shots());

    //the signal is used for asynchronous purposes (in UI classes), and the return value for synchronous (in non-UI classes)
    if(!signalsBlocked())
        emit ftDone(spectrum,id);

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

    Ft out;
    out.reserve(r.size() + d.size());
    out.setLoFreq(r.loFreqMHz());
    out.setSpacing(r.xSpacing());

    if(qFuzzyCompare(r.loFreqMHz(),d.loFreqMHz()))
    {
        out.setX0(r.minFreqMHz());
        for(int i=0; i<r.size() && i<d.size(); i++)
            out.append(r.at(i) - d.at(i));
    }
    else
    {
        auto [drs,f0] = resample(r.minFreqMHz(),r.xSpacing(),d);
        clearSplineMemory();

        out.setX0(qMin(r.minFreqMHz(),f0));

        int offset = static_cast<int>((f0-r.minFreqMHz())/r.xSpacing());
        if(offset < 0) // the ref is at a higher frequency; start from the diff
        {
            for(int i=0; (i+offset)<r.size(); ++i)
                out.append(r.value(i+offset) - drs.value(i,0.0));
        }
        else
        {
            for(int i=0; (i-offset)<d.size(); ++i)
                out.append(r.value(i) - drs.value(i-offset,0.0));
        }
    }

    out.squeeze();
    emit ftDiffDone(out);

}

Ft FtWorker::processSideband(const FidList fl, const FtWorker::FidProcessingSettings &settings, RfConfig::Sideband sb,double minFreq, double maxFreq)
{
    //this will FT all of the FIDs and resample them on a common grid
    QList<Ft> ftList = makeSidebandList(fl,settings,sb,minFreq,maxFreq);

    if(ftList.isEmpty())
    {
//        emit ftDone(Ft(),d_id);
        return Ft();
    }

    if(ftList.size() == 1)
    {
//        emit ftDone(ftList.constFirst(),d_id);
        return ftList.constFirst();
    }

//    QVector<int> indices;
//    indices.resize(ftList.size());

//    Ft out(0,0.0);
//    out.reserve(ftList.size()*ftList.constFirst().size());

//    //want to make sure frequency increases monotonically as we iterate through fidlist
//    //Fts ALWAYS go from low frequency to high
//    if(ftList.constFirst().loFreqMHz() > ftList.constLast().loFreqMHz())
//        std::reverse(ftList.begin(),ftList.end());

//    while(true)
//    {
//        double thisPointY = 0.0;
//        double thisPointX = 0.0;
//        double numPoints = 0.0;

//        for(int i=0; i<indices.size(); i++)
//        {
//            if(indices.at(i) < ftList.at(i).size())
//            {
//                thisPointX = ftList.at(i).at(indices.at(i)).x();
//                break;
//            }
//        }

//        for(int i=0; i<indices.size(); i++)
//        {
//            if(indices.at(i) < ftList.at(i).size())
//            {
//                if(qAbs(thisPointX - ftList.at(i).at(indices.at(i)).x()) < ftList.at(i).xSpacing())
//                {
//                    double y = ftList.at(i).at(indices.at(i)).y();
//                    if(y>0)
//                    {
//                        thisPointY += log10(y);
//                        numPoints += 1.0;
//                    }
//                    indices[i]++;
//                }
//            }
//        }

//        if(numPoints < 1.0)
//            out.append(QPointF(thisPointX,thisPointY));
//        else
//            out.append(QPointF(thisPointX,pow(10.0,thisPointY/numPoints)));

//        bool done = true;
//        for(int i=0; i<indices.size(); i++)
//        {
//            if(indices.at(i) < ftList.at(i).size())
//            {
//                done = false;
//                break;
//            }
//        }

//        if(done)
//            break;
//    }

//    emit ftDone(out,d_id);
//    return out;
}

void FtWorker::processBothSidebands(const FidList fl, const FtWorker::FidProcessingSettings &settings, double minFreq, double maxFreq)
{
//    blockSignals(true);
//    Ft upper = processSideband(fl,settings,RfConfig::UpperSideband,minFreq,maxFreq);
//    Ft lower = processSideband(fl,settings,RfConfig::LowerSideband,minFreq,maxFreq);
//    blockSignals(false);

//    Ft out(0,0.0);
//    if(upper.size() < 2)
//    {
//        if(lower.size() < 2)
//        {
//            emit ftDone(out,d_id);
//            return;
//        }
//        else
//        {
//            emit ftDone(lower,d_id);
//            return;
//        }
//    }
//    else
//    {
//        if(lower.size() < 2)
//        {
//            emit ftDone(upper,d_id);
//            return;
//        }
//    }

//    out.reserve(upper.size() + lower.size());

//    int li=0, ui=0;

//    while(true)
//    {
//        QPointF pt;

//        if(li < lower.size())
//        {
//            pt = lower.at(li);
//            if(qAbs(pt.x()-upper.at(ui).x()) < lower.xSpacing())
//            {
//                pt.setY((pt.y() + upper.at(ui).y())/2.0);
//                ui++;
//            }

//            li++;
//        }
//        else
//        {
//            pt = upper.at(ui);
//            ui++;
//        }
//        out.append(pt);

//        if(ui < upper.size() || li < lower.size())
//            continue;

//        break;

//    }

//    emit ftDone(out,d_id);

}

QList<Ft> FtWorker::makeSidebandList(const FidList fl, const FidProcessingSettings &settings, RfConfig::Sideband sb, double minFreq, double maxFreq)
{
    if(fl.isEmpty())
        return QList<Ft>();

    Fid f = fl.constFirst();
    f.setSideband(sb);

    QList<Ft> out;

    bool sigsBlocked = signalsBlocked();
    blockSignals(true);
    Ft ft1 = doFT(f,settings);
    ft1.trim(minFreq,maxFreq);
    out << ft1;

    double f0 = 0.0;
    double sp = -1.0;


    for(int i=1; i<fl.size(); i++)
    {
        f = fl.at(i);
        f.setSideband(sb);
        ft1 = doFT(f,settings);
        if(!ft1.isEmpty())
        {
            ft1.trim(minFreq,maxFreq);
            if(sp < 0.0)
            {
                f0 = ft1.loFreqMHz();
                sp = ft1.xSpacing();
            }
            out << ft1;
        }
        else
        {
            auto rsft = resample(f0,sp,ft1);
//            out << rsft;
        }
    }
    blockSignals(sigsBlocked);

    return out;

}

FtWorker::FilterResult FtWorker::filterFid(const Fid fid, const FidProcessingSettings &settings)
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

    double min = data.at(si);
    double max = min;
    QReadLocker l(pu_winfLock.get());
    if(settings.windowFunction != d_lastWinf || d_lastWinSize != n)
    {
        l.unlock();
        QWriteLocker l2(pu_winfLock.get());
        makeWinf(n,settings.windowFunction);
        l2.unlock();
        l.relock();
    }
    for(int i=0; i<data.size(); i++)
    {
        if(i < si)
            continue;

        if(i > ei)
            break;


        double d = data.at(i);
        if(settings.windowFunction != None)
            d*=d_winf.at(i-si);

        out[i] = d;
        min = qMin(d,min);
        max = qMax(d,max);
    }

    if(settings.zeroPadFactor > 0 && settings.zeroPadFactor <= 2)
    {
        int filledSize = Analysis::nextPowerOf2(data.size() * (1 << settings.zeroPadFactor));
        if(out.size() != filledSize)
            out.resize(filledSize);
    }

    return {out,min,max};

}

QPair<QVector<double>,double> FtWorker::resample(double f0, double spacing, const Ft ft)
{
    if(ft.isEmpty() || ft.size() < 2 || spacing == 0.0)
        return {};

    if(qFuzzyCompare(f0,ft.xFirst()) && qFuzzyCompare(spacing,ft.xSpacing()))
        return {};


    double minF = ft.minFreqMHz();

    //find sample point closest to minf
    double firstPt = f0 + round((minF-f0)/spacing)*spacing;

    int numPoints = ft.size();

    //set up spline object with FT data
    auto xd = ft.xData();
    auto yd = ft.yData();

    //allocate or reallocate gsl_spline object
    QReadLocker l(pu_splineLock.get());
    if(ft.size() != d_numSplinePoints)
    {
        l.unlock();
        QWriteLocker l2(pu_splineLock.get());
        d_numSplinePoints = ft.size();

        if(p_spline != nullptr)
            gsl_spline_free(p_spline);

        p_spline = gsl_spline_alloc(gsl_interp_cspline,d_numSplinePoints);

        if(p_accel != nullptr)
        {
            gsl_interp_accel_free(p_accel);
            p_accel = gsl_interp_accel_alloc();
        }
        l2.unlock();
        l.relock();
    }


    gsl_spline_init(p_spline,xd.constData(),yd.constData(),d_numSplinePoints);

    QVector<double> out;
    out.reserve(numPoints);

    for(int i=0; i<numPoints; ++i)
    {
        double x = firstPt + static_cast<double>(i)*spacing;
        double y = gsl_spline_eval(p_spline,x,p_accel);
        if(isnan(y))
            y = 0.0;
        out.append(y);
    }

    return {out,firstPt};

}

void FtWorker::makeWinf(int n, FtWindowFunction f)
{
    if(d_winf.size() != n)
        d_winf.resize(n);

    switch(f)
    {
    case Bartlett:
        winBartlett(n);
        break;
    case Blackman:
        winBlackman(n);
        break;
    case BlackmanHarris:
        winBlackmanHarris(n);
        break;
    case Hamming:
        winHamming(n);
        break;
    case Hanning:
        winHanning(n);
        break;
    case KaiserBessel:
        winKaiserBessel(n,14.0);
        break;
    case None:
    default:
        d_winf.fill(1.0);
        break;
    }

    d_lastWinf = f;
    d_lastWinSize = n;
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

void FtWorker::clearSplineMemory()
{
    QWriteLocker l(pu_splineLock.get());
    gsl_spline_free(p_spline);
    gsl_interp_accel_free(p_accel);
    p_spline = nullptr;
    p_accel = nullptr;

    d_numSplinePoints = -1;
}
