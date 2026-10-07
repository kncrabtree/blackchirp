#include <data/lif/fcutunesweep.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include <Eigen/Dense>

using namespace Qt::StringLiterals;
using namespace BC::FcuTune;

FcuTuneSweep::FcuTuneSweep(const Settings &settings, double startTrim) :
    d_settings(settings), d_startTrim(startTrim)
{
    d_settings.points = std::max(3,d_settings.points);
    d_settings.waveformsPerPoint = std::max(1,d_settings.waveformsPerPoint);
    d_settings.discardPerPoint = std::max(0,d_settings.discardPerPoint);
    if(!(d_settings.halfWidth > 0.0))
        d_settings.halfWidth = 1.0;
    d_settings.direction = d_settings.direction < 0 ? -1 : 1;

    auto n = d_settings.points;
    auto step = 2.0*d_settings.halfWidth/static_cast<double>(n-1);
    d_trims.reserve(static_cast<std::size_t>(n));
    for(int i=0; i<n; i++)
        d_trims.push_back(d_startTrim - d_settings.halfWidth + step*static_cast<double>(i));
    if(d_settings.direction < 0)
        std::reverse(d_trims.begin(),d_trims.end());

    d_points.resize(d_trims.size());
}

double FcuTuneSweep::currentTrim() const
{
    auto i = std::min(d_index,static_cast<int>(d_trims.size())-1);
    return d_trims.at(static_cast<std::size_t>(i));
}

bool FcuTuneSweep::addWaveform(double refIntegral, bool clipped)
{
    if(isComplete())
        return true;

    auto &p = d_points[static_cast<std::size_t>(d_index)];
    p.seen++;
    if(p.seen <= d_settings.discardPerPoint)
        return false;

    if(p.accepted < d_settings.waveformsPerPoint)
    {
        p.accepted++;
        p.sum += refIntegral;
        p.sumSq += refIntegral*refIntegral;
        d_clipped |= clipped;
    }

    return p.accepted >= d_settings.waveformsPerPoint;
}

bool FcuTuneSweep::advance()
{
    if(!isComplete())
        d_index++;
    return !isComplete();
}

int FcuTuneSweep::perMilComplete() const
{
    auto total = static_cast<double>(d_trims.size())*d_settings.waveformsPerPoint;
    double done = 0.0;
    for(const auto &p : d_points)
        done += p.accepted;

    return std::clamp(static_cast<int>(1000.0*done/total),0,1000);
}

std::pair<double, double> FcuTuneSweep::pointStats(int i) const
{
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
    if(i < 0 || i >= static_cast<int>(d_points.size()))
        return {nan,nan};

    const auto &p = d_points[static_cast<std::size_t>(i)];
    if(p.accepted < 1)
        return {nan,nan};

    auto n = static_cast<double>(p.accepted);
    auto mean = p.sum/n;
    if(p.accepted < 2)
        return {mean,0.0};

    auto var = std::max(0.0,(p.sumSq - n*mean*mean)/(n-1.0));
    return {mean,std::sqrt(var/n)};
}

Result FcuTuneSweep::result() const
{
    std::vector<double> means, errs;
    means.reserve(d_points.size());
    errs.reserve(d_points.size());
    for(int i=0; i<static_cast<int>(d_points.size()); i++)
    {
        auto [m,e] = pointStats(i);
        means.push_back(m);
        errs.push_back(e);
    }

    auto out = fitPeak(d_trims,means,errs,d_settings.minContrast);
    if(d_clipped)
    {
        out.status = Status::Clipped;
        out.message = u"The reference signal saturated during the sweep (digitizer full scale or the "
                       "saturation level); attenuate the light on the photodiode or reduce its gain."_s;
    }

    return out;
}

Result FcuTuneSweep::fitPeak(const std::vector<double> &trims, const std::vector<double> &means,
                             const std::vector<double> &stdErrs, double minContrast)
{
    Result out;
    out.trims = trims;
    out.means = means;
    out.stdErrs = stdErrs;
    out.center = std::numeric_limits<double>::quiet_NaN();
    out.centerUncertainty = std::numeric_limits<double>::quiet_NaN();

    auto n = trims.size();
    if(n < 3 || means.size() != n || stdErrs.size() != n)
    {
        out.status = Status::InsufficientData;
        out.message = u"At least 3 points with matching statistics are required."_s;
        return out;
    }

    for(std::size_t i=0; i<n; i++)
    {
        if(!std::isfinite(trims[i]) || !std::isfinite(means[i]))
        {
            out.status = Status::InsufficientData;
            out.message = u"Point %1 has no accepted waveforms."_s.arg(i+1);
            return out;
        }
    }

    // Work in ascending trim order regardless of visiting order.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&trims](auto a, auto b){ return trims[a] < trims[b]; });

    // Orient so the peak is a maximum: a negative-going reference pulse
    // integrates to a negative signal.
    auto total = std::accumulate(means.begin(),means.end(),0.0);
    double sign = total < 0.0 ? -1.0 : 1.0;

    std::vector<double> x(n), y(n), e(n);
    for(std::size_t i=0; i<n; i++)
    {
        x[i] = trims[order[i]];
        y[i] = sign*means[order[i]];
        e[i] = std::isfinite(stdErrs[order[i]]) ? std::abs(stdErrs[order[i]]) : 0.0;
    }

    auto k = static_cast<std::size_t>(std::distance(y.begin(),std::max_element(y.begin(),y.end())));
    auto ymax = y[k];
    auto ymin = *std::min_element(y.begin(),y.end());

    if(!(ymax > 0.0))
    {
        out.status = Status::LowContrast;
        out.contrast = 0.0;
        out.message = u"No reference signal detected; check the reference channel and gate."_s;
        return out;
    }

    out.contrast = (ymax - ymin)/ymax;
    if(out.contrast < minContrast)
    {
        out.status = Status::LowContrast;
        out.message = u"Reference signal varies by only %1% across the window (minimum %2%); "
                       "widen the window."_s
                          .arg(100.0*out.contrast,0,'f',1).arg(100.0*minContrast,0,'f',1);
        return out;
    }

    if(k == 0 || k == n-1)
    {
        out.status = Status::PeakAtEdge;
        out.center = x[k];
        out.message = u"Maximum is at the edge of the window (trim %1); the peak may lie outside it. "
                       "Re-center the window there or widen it."_s.arg(x[k],0,'f',0);
        return out;
    }

    // Contiguous run at or above half maximum, at least the maximum and its
    // two neighbors.
    auto lo = k, hi = k;
    while(lo > 0 && y[lo-1] >= 0.5*ymax)
        lo--;
    while(hi < n-1 && y[hi+1] >= 0.5*ymax)
        hi++;
    lo = std::min(lo,k-1);
    hi = std::max(hi,k+1);

    for(auto i=lo; i<=hi; i++)
    {
        if(!(y[i] > 0.0))
        {
            out.status = Status::FitFailed;
            out.message = u"The peak is narrower than the trim step; reduce the window or add points."_s;
            return out;
        }
    }

    // Weighted parabola in ln(y), in coordinates centered on the maximum and
    // scaled by the fitted span for conditioning. sigma(ln y) = sigma(y)/y;
    // when any point lacks an error estimate, fall back to equal weights and
    // estimate the scale from the residuals.
    auto m = static_cast<int>(hi - lo + 1);
    auto scale = std::max(x[hi] - x[lo],std::numeric_limits<double>::min());
    bool haveErrors = true;
    for(auto i=lo; i<=hi; i++)
        haveErrors &= e[i] > 0.0;

    Eigen::MatrixXd A(m,3);
    Eigen::VectorXd b(m), w(m);
    for(int r=0; r<m; r++)
    {
        auto i = lo + static_cast<std::size_t>(r);
        auto u = (x[i] - x[k])/scale;
        A(r,0) = 1.0;
        A(r,1) = u;
        A(r,2) = u*u;
        b(r) = std::log(y[i]);
        w(r) = haveErrors ? (y[i]*y[i])/(e[i]*e[i]) : 1.0;
    }

    Eigen::MatrixXd N = A.transpose()*w.asDiagonal()*A;
    Eigen::VectorXd rhs = A.transpose()*w.asDiagonal()*b;
    Eigen::FullPivLU<Eigen::MatrixXd> lu(N);
    if(!lu.isInvertible())
    {
        out.status = Status::FitFailed;
        out.message = u"Peak fit is singular."_s;
        return out;
    }

    Eigen::VectorXd p = lu.solve(rhs);
    Eigen::MatrixXd cov = lu.inverse();
    if(!haveErrors)
    {
        if(m > 3)
        {
            Eigen::VectorXd res = A*p - b;
            cov *= res.squaredNorm()/static_cast<double>(m-3);
        }
        else
            cov.setConstant(std::numeric_limits<double>::quiet_NaN());
    }

    auto c1 = p(1), c2 = p(2);
    if(!(c2 < 0.0))
    {
        out.status = Status::FitFailed;
        out.message = u"Peak fit has no maximum (no negative curvature)."_s;
        return out;
    }

    auto u0 = -c1/(2.0*c2);
    auto center = x[k] + u0*scale;
    if(center < x[lo] || center > x[hi])
    {
        out.status = Status::FitFailed;
        out.message = u"Fitted peak (trim %1) falls outside the fitted points."_s.arg(center,0,'f',0);
        return out;
    }

    // Propagate the (b, c) covariance through u0 = -b/(2c).
    Eigen::Vector2d J(-1.0/(2.0*c2), c1/(2.0*c2*c2));
    Eigen::Matrix2d C = cov.block<2,2>(1,1);
    auto var = J.dot(C*J);

    out.status = Status::Success;
    out.center = center;
    out.centerUncertainty = std::isfinite(var) ? std::sqrt(std::max(0.0,var))*scale
                                               : std::numeric_limits<double>::quiet_NaN();
    out.message = u"Peak at trim %1."_s.arg(center,0,'f',0);
    return out;
}
