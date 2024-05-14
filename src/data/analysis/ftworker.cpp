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

Ft FtWorker::doFT(const FidList fl, const FidProcessingSettings &settings, int frame, int id, bool doubleSideband)
{
    if(fl.isEmpty() || frame >= fl.size())
    {
        if(id > -1)
        {
            emit fidDone({},1.0,0.0,0.0,0,id);
            emit ftDone(Ft(), id);
        }
        return Ft();
    }

    //frame -1 means average all frames
    Fid fid;
    if(frame < 0)
    {
        fid = fl.at(0);
        for(int i=1; i<fl.size(); i++)
            fid += fl.at(i);
    }
    else
        fid = fl.value(frame,Fid());

    if(fid.size() < 2)
    {
        if(id > -1)
        {
            emit fidDone({},1.0,0.0,0.0,0,id);
            emit ftDone(Ft(), id);
        }
        return Ft();
    }

    double rawSize = static_cast<double>(fid.size());

    //first, apply any filtering that needs to be done
    auto fidResult = filterFid(fid,settings);
    if(id > -1)
        emit fidDone(fidResult.fid,fid.spacing()*1e6,fidResult.min,fidResult.max,fid.shots(),id);
    auto fftData = fidResult.fid;
    auto s = fftData.size();


    auto spacing = fid.spacing()*1.0e6;
    auto probe = fid.probeFreq();
    double ftSpacing = 1.0/static_cast<double>(s)/spacing;
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
    if(id>-1)
        emit ftDone(spectrum,id);

    return spectrum;
}

void FtWorker::doFtDiff(const FidList refList, const FidList diffList, int refFrame, int diffFrame, const FidProcessingSettings &settings)
{
    if(refList.size() != diffList.size())
        return;

    for(int i=0; i<refList.size(); i++)
    {
        auto &ref = refList.at(i);
        auto &diff = diffList.at(i);
        if(ref.size() != diff.size() || ref.sideband() != diff.sideband())
            return;

        if(!qFuzzyCompare(ref.spacing(),diff.spacing()))
            return;
    }


    Ft r = doFT(refList,settings,refFrame);
    Ft d = doFT(diffList,settings,diffFrame);

    Ft out;
    out.reserve(r.size() + d.size());
    out.setLoFreq(r.loFreqMHz());
    out.setSpacing(r.xSpacing());
    out.setNumShots(r.shots() + d.shots());

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

void FtWorker::processSideband2(const FtWorker::SidebandProcessingData &d, const FtWorker::FidProcessingSettings &settings)
{
    if(d.currentIndex >= d.totalFids)
        return;

    if(d.currentIndex == 0)
    {
        d_loScanData = {};

        double minProbeFreq = d.loRange.first, maxProbeFreq = d.loRange.second;
        double bandwidth = 1.0/2.0/d.fl.constFirst().spacing()/1e6;
        int s = d.fl.constFirst().size();
        if(settings.zeroPadFactor > 0 && settings.zeroPadFactor <= 2)
            s = Analysis::nextPowerOf2(s * (1 << settings.zeroPadFactor));
        int ftPoints = s/2+1;
        d_loScanData.ftSpacing = bandwidth/(ftPoints-1);

        if(d.doubleSideband)
        {
            auto ftRange = maxProbeFreq - minProbeFreq + 2*d.maxOffset;
            d_loScanData.ftPoints = static_cast<uint>(ceil(ftRange/d_loScanData.ftSpacing))+1;
            d_loScanData.ftXRange.first = minProbeFreq-d.maxOffset;
            d_loScanData.ftXRange.second = d_loScanData.ftXRange.first + d_loScanData.ftSpacing*d_loScanData.ftPoints;
        }
        else if(d.sideband == RfConfig::LowerSideband)
        {
            auto ftRange = maxProbeFreq - minProbeFreq + (d.maxOffset-d.minOffset);
            d_loScanData.ftPoints = static_cast<uint>(ceil(ftRange/d_loScanData.ftSpacing))+1;
            d_loScanData.ftXRange.first = minProbeFreq - d.maxOffset;

            d_loScanData.ftXRange.second = d_loScanData.ftXRange.first + d_loScanData.ftSpacing*d_loScanData.ftPoints;
        }
        else
        {
            auto ftRange = maxProbeFreq - minProbeFreq + (d.maxOffset-d.minOffset);
            d_loScanData.ftPoints = static_cast<uint>(ceil(ftRange/d_loScanData.ftSpacing))+1;
            d_loScanData.ftXRange.first = minProbeFreq + d.minOffset;

            d_loScanData.ftXRange.second = d_loScanData.ftXRange.first + d_loScanData.ftSpacing*d_loScanData.ftPoints;
        }

        d_loScanData.ftData.resize(d_loScanData.ftPoints);
        d_loScanData.counts.clear();
        d_loScanData.counts.resize(d_loScanData.ftPoints);
    }

    if(!d.fl.isEmpty() && d.fl.constFirst().shots() > 0)
    {
        auto fl = d.fl;
        if(!d.doubleSideband)
        {
            for(auto &fid : fl)
                fid.setSideband(d.sideband);
        }

        auto ft = doFT(fl,settings,d.frame,-1,d.doubleSideband);
        auto pf = fl.constFirst().probeFreq();
        if(d.doubleSideband)
            ft.trim(pf-d.maxOffset,pf+d.maxOffset);
        else if(d.sideband == RfConfig::UpperSideband)
            ft.trim(pf+d.minOffset,pf+d.maxOffset);
        else
            ft.trim(pf-d.maxOffset,pf-d.minOffset);


        auto index = d_loScanData.indexOf(ft.minFreqMHz());
        auto xx = d_loScanData.relDistance(ft.xAt(0));
        for(uint i=0; i<(uint)ft.size() && i+index < d_loScanData.ftPoints; i++)
        {
            //linear interpolation onto total grid, averaging in 0 for outermost points.
            double yint = ft.at(i);
            if(xx > 0.0)
            {
                if(i == 0)
                    yint = xx*ft.at(i);
                else if(i+1 == (uint)ft.size())
                    yint = ft.at(i)*(1-xx);
                else
                    yint = ft.at(i) + (ft.at(i+1)-ft.at(i))*xx;
            }

            //do average
            if(d_loScanData.counts.at(i+index) == 0)
            {
                d_loScanData.ftData[i+index] = yint;
                d_loScanData.counts[i+index] = ft.shots();
            }
            else
            {
                auto s1 = d_loScanData.counts.at(i+index);
                auto y1 = d_loScanData.ftData.at(i+index);
                auto s2 = ft.shots();
                switch(d.dcMethod)
                {
                case Harmonic_Mean:
                    d_loScanData.ftData[i+index] = (s1+s2)/(s1/y1+s2/yint);
                    break;
                case Geometric_Mean:
                    d_loScanData.ftData[i+index] = exp((s1*log(y1) + s2*log(yint))/(s1+s2));
                    break;
                }

                d_loScanData.counts[i+index] += s2;
            }
        }

        d_loScanData.totalShots += ft.shots();
    }

    if(d.currentIndex + 1 >= d.totalFids)
    {
        Ft out(d_loScanData.ftPoints,d_loScanData.ftXRange.first,d_loScanData.ftSpacing,d_loScanData.ftXRange.first);
        double yMin = 0.0, yMax = 0.0;
        for(int i=0; i<d_loScanData.ftData.size(); i++)
        {
            yMin = qMin(yMin,d_loScanData.ftData.at(i));
            yMax = qMax(yMax,d_loScanData.ftData.at(i));
        }
        out.setData(d_loScanData.ftData,yMin,yMax);
        out.setNumShots(d_loScanData.totalShots);
        emit sidebandDone(out);
    }

}

void FtWorker::processSideband(const FtWorker::SidebandProcessingData &d, const FtWorker::FidProcessingSettings &settings)
{
    if(d.currentIndex >= d.totalFids)
        return;

    if(d.currentIndex == 0)
    {
        d_workingSidebandFt = Ft();
        d_sidebandIndices.clear();
        clearSplineMemory();
    }

    auto fl = d.fl;

    if(!d.doubleSideband)
    {
        for(auto &fid : fl)
            fid.setSideband(d.sideband);
    }

    if(!d.fl.isEmpty())
    {
        auto ft = doFT(fl,settings,d.frame,-1,d.doubleSideband);
        if(d.minOffset > 0.0 || d.maxOffset < (ft.maxFreqMHz()-ft.minFreqMHz()))
            ft.trim(d.minOffset,d.maxOffset);

        if(d.currentIndex == 0)
        {
            d_workingSidebandFt = ft;
            d_sidebandIndices.emplace(0,1);
            d_sidebandIndices.emplace(ft.size()-1,-1);
        }
        else
        {
            auto [rhs,f0] = resample(d_workingSidebandFt.xFirst(),d_workingSidebandFt.xSpacing(),ft);
                    int offset = static_cast<int>((f0-d_workingSidebandFt.minFreqMHz())/d_workingSidebandFt.xSpacing());

                    //compute size of new working FT
                    int leftIndex = qMin(offset,0);
                    int rightIndex = qMax(offset+rhs.size()-1,d_workingSidebandFt.size()-1);
                    int newSize = rightIndex - leftIndex + 1;

                    QVector<double> lhs = d_workingSidebandFt.yData();
                    QVector<double> newData;
                    newData.reserve(newSize);
                    bool origlhs = true;

                    if(offset < 0)
            {
                //need to rebuild working Ft with new minimum frequency
                //adjust indices to new reference
                auto copy = d_sidebandIndices;
                d_sidebandIndices.clear();
                for(auto &[key,val] : copy)
                    d_sidebandIndices.emplace(key-offset,val);

                d_sidebandIndices.emplace(0,1);
                d_sidebandIndices.emplace(ft.size()-1,-1);

                d_workingSidebandFt.setX0(f0);
                qSwap(lhs,rhs);
                offset = -offset;
                origlhs = false;
            }
            else
            {
                d_sidebandIndices.emplace(offset,1);
                d_sidebandIndices.emplace(offset+ft.size()-1,-1);
            }

            //at this point, the first point in lhs is index 0, and the first point in rhs is index offset.
            //using the working sideband indices, create a data structure with current counts (0 where data will be new)
            int currentCount = 0;

            //loop over new points. If count is 0, append lhs if count is < offet, rhs if count > offset
            //If count is 1 or greater, do geometric mean if lhs and rhs are both > 0; arithmetic mean otherwise
            //use the counts structure to determine the current count
            double yMin = 0.0;
            double yMax = 0.0;
            int i = 0;

            //this loop is structured to make effective use of branch prediction
            //most of the if statements in the while statements will evaluate the same way
            //in each iteration of the outer for loop. The exception is the code that
            //detects if either point is a 0, but that condition should be rare.
            for(auto it = d_sidebandIndices.cbegin(); it != d_sidebandIndices.cend(); ++it)
            {
                //possible that there are duplicate indices if ane segment starts when another stops
                while(i > it->first && it != d_sidebandIndices.cend())
                {
                    currentCount += it->second;
                    ++it;
                }


                double thisCount = currentCount;
                double totalCount = thisCount+1.0;
                double ratio = thisCount/totalCount;
                if(it == d_sidebandIndices.cend())
                {
                    while(i < newSize-1)
                    {
                        auto d = rhs.at(i+offset);
                        yMin = qMin(yMin,d);
                        yMax = qMax(yMax,d);
                        newData.append(d);
                        ++i;
                    }
                    break;
                }

                while(i < it->first)
                {
                    if(i >= lhs.size())
                    {
                        auto val = rhs.value(i-offset);
                        yMin = qMin(yMin,val);
                        yMax = qMax(yMax,val);
                        newData.append(val);
                        ++i;
                    }
                    else
                    {
                        auto d = lhs.at(i);
                        if(it->second == 0 || i < offset)
                        {
                            //no points yet at this index; store the current value and move on
                            yMin = qMin(yMin,d);
                            yMax = qMax(yMax,d);
                            newData.append(d);
                            ++i;
                        }
                        else
                        {
                            auto d2 = rhs.at(i-offset);
                            if(d <= 0.0 || d2 <=0.0)
                            {
                                //compute arithmetic mean
                                double val;
                                if(origlhs)
                                    val = ratio*d + ratio/totalCount;
                                else
                                    val = d/totalCount + ratio*d2;
                                yMin = qMin(yMin,val);
                                yMax = qMax(yMax,val);
                                newData.append(val);
                                ++i;
                            }
                            else
                            {
                                //compute geometric mean
                                double val;
                                if(origlhs)
                                    val = pow(10.0,ratio*log10(d) + log10(d2)/totalCount);
                                else
                                    val = pow(10.0,log10(d)/totalCount + ratio*log10(d2));
                                yMin = qMin(yMin,val);
                                yMax = qMax(yMax,val);
                                newData.append(val);
                                ++i;
                            }
                        }
                    }
                }

                currentCount += it->second;
            }

            //if points remain, fill them in
            while(i < newSize-1)
            {
                auto d = rhs.at(i+offset);
                yMin = qMin(yMin,d);
                yMax = qMax(yMax,d);
                newData.append(d);
                ++i;
            }

            d_workingSidebandFt.setData(newData,yMin,yMax);
            d_workingSidebandFt.setNumShots(d_workingSidebandFt.shots() + ft.shots());
        }
    }

    if(d.currentIndex + 1 >= d.totalFids)
    {
        emit sidebandDone(d_workingSidebandFt);
        d_workingSidebandFt = Ft();
        clearSplineMemory();
        d_sidebandIndices.clear();
    }

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

    double avg = 0.0;

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

        avg = sum/static_cast<double>(n);
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


        double d = settings.removeDC ? data.at(i)-avg : data.at(i);

        if(settings.expFilter > 0.0)
            d*=exp(-static_cast<double>(i-si)*fid.spacing()/(settings.expFilter/1e6));

        if(settings.windowFunction != None)
            d*=d_winf.at(i-si);

        out[i] = d;
        if(i==si)
        {
            min = d;
            max = d;
        }
        else
        {
            min = qMin(d,min);
            max = qMax(d,max);
        }
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
