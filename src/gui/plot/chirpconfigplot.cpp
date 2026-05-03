#include "chirpconfigplot.h"
#include "curvefactory.h"

#include <QMouseEvent>
#include <QMenu>

#include <qwt6/qwt_legend.h>

#include <gui/plot/blackchirpplotcurve.h>
#include <data/experiment/chirpconfig.h>

ChirpConfigPlot::ChirpConfigPlot(QWidget *parent) : ZoomPanPlot(BC::Key::chirpPlot,parent)
{
    setPlotAxisTitle(QwtPlot::xBottom,QString::fromUtf16(u"Time (μs)"));
    setPlotAxisTitle(QwtPlot::yLeft,QString("Chirp (Normalized)"));

    // Disable QwtPlot's automatic memory management
    setAutoDelete(false);

    // Create chirp waveform curve using CurveFactory
    p_chirpCurve = CurveFactory::createStandardCurve<BlackchirpPlotCurve>(BC::Key::chirpCurve);
    attachCurve(p_chirpCurve.get());

    insertLegend(new QwtLegend(this));
}

ChirpConfigPlot::~ChirpConfigPlot()
{
    // All items are managed by unique_ptr and will be automatically cleaned up
}

void ChirpConfigPlot::newChirp(const ChirpConfig cc)
{
    if(cc.chirpList().isEmpty())
        return;

    bool as = p_chirpCurve->curveData().isEmpty();

    auto chirpData = cc.getChirpMicroseconds();
    const auto &channels = cc.markerChannels();
    int numChannels = channels.size();

    // Resize marker curves: destroying a unique_ptr detaches the curve from the plot
    while(d_markerCurves.size() > static_cast<std::size_t>(numChannels))
        d_markerCurves.pop_back();

    while(d_markerCurves.size() < static_cast<std::size_t>(numChannels))
    {
        int idx = static_cast<int>(d_markerCurves.size());
        auto curve = CurveFactory::createStandardCurve<BlackchirpPlotCurve>(
            QString("Marker%1").arg(idx));
        attachCurve(curve.get());
        d_markerCurves.push_back(std::move(curve));
    }

    if(chirpData.isEmpty())
    {
        p_chirpCurve->setCurveData(QVector<QPointF>());
        for(auto &c : d_markerCurves)
            c->setCurveData(QVector<QPointF>());
        autoScale();
        return;
    }

    if(as)
        autoScale();

    // Build per-channel curve data
    QVector<QVector<QPointF>> markerData(numChannels);
    double lead = cc.leadTimeUs();

    for(int i = 0; i < cc.numChirps(); i++)
    {
        double intervalStart = cc.chirpInterval() * static_cast<double>(i);
        double chirpStart = intervalStart + lead;
        double chirpEnd = chirpStart + cc.chirpDurationUs(i);

        for(int ch = 0; ch < numChannels; ++ch)
        {
            const auto &m = channels.at(ch);
            if(!m.enabled)
                continue;

            double mStart = chirpStart + m.startTime;
            double mEnd = chirpEnd + m.endTime;
            auto &pts = markerData[ch];
            pts.append(QPointF(intervalStart, 0.0));
            pts.append(QPointF(mStart, 0.0));
            pts.append(QPointF(mStart, 1.0));
            pts.append(QPointF(mEnd, 1.0));
            pts.append(QPointF(mEnd, 0.0));
        }
    }

    p_chirpCurve->setCurveData(chirpData);

    for(int ch = 0; ch < numChannels; ++ch)
    {
        const auto &m = channels.at(ch);

        QString label = QString("Marker %1").arg(ch);
        switch(m.role)
        {
        case MarkerRole::Protection: label += " (Protection)"; break;
        case MarkerRole::Gate:       label += " (Gate)";       break;
        case MarkerRole::Trigger:    label += " (Trigger)";    break;
        default: break;
        }

        d_markerCurves.at(ch)->setTitle(label);
        d_markerCurves.at(ch)->setCurveData(markerData.at(ch));
    }

    replot();
}
