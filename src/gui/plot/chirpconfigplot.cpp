#include "chirpconfigplot.h"

#include <QMouseEvent>
#include <QMenu>

#include <qwt6/qwt_legend.h>

#include <gui/plot/blackchirpplotcurve.h>
#include <data/experiment/chirpconfig.h>

ChirpConfigPlot::ChirpConfigPlot(QWidget *parent) : ZoomPanPlot(BC::Key::chirpPlot,parent)
{
    setPlotAxisTitle(QwtPlot::xBottom,QString::fromUtf16(u"Time (μs)"));
    setPlotAxisTitle(QwtPlot::yLeft,QString("Chirp (Normalized)"));

    p_chirpCurve = new BlackchirpPlotCurve(BC::Key::chirpCurve);
    p_chirpCurve->attach(this);

    p_ampEnableCurve= new BlackchirpPlotCurve(BC::Key::ampCurve);
    p_ampEnableCurve->attach(this);

    p_protectionCurve = new BlackchirpPlotCurve(BC::Key::protCurve);
    p_protectionCurve->attach(this);

    insertLegend(new QwtLegend(this));

}

ChirpConfigPlot::~ChirpConfigPlot()
{
    detachItems();
}

void ChirpConfigPlot::newChirp(const ChirpConfig cc)
{
    if(cc.chirpList().isEmpty())
        return;

    bool as = p_chirpCurve->curveData().isEmpty();

    auto chirpData = cc.getChirpMicroseconds();
    if(chirpData.isEmpty())
    {
        p_chirpCurve->setCurveData(QVector<QPointF>());
        p_ampEnableCurve->setCurveData(QVector<QPointF>());
        p_protectionCurve->setCurveData(QVector<QPointF>());
        autoScale();
        return;
    }

    if(as)
        autoScale();

    QVector<QPointF> ampData, protectionData;

    for(int i=0; i<cc.numChirps(); i++)
    {
        double segmentStartTime = cc.chirpInterval()*static_cast<double>(i);
        double twtEnableTime = segmentStartTime + cc.preChirpProtectionDelay();
        double chirpEndTime = twtEnableTime + cc.preChirpGateDelay() + cc.chirpDurationUs(i);
        double twtEndTime = chirpEndTime + cc.postChirpGateDelay();
        double protectionEndTime = chirpEndTime + cc.postChirpProtectionDelay();

        //build protection data
        protectionData.append(QPointF(segmentStartTime,0.0));
        protectionData.append(QPointF(segmentStartTime,1.0));
        protectionData.append(QPointF(protectionEndTime,1.0));
        protectionData.append(QPointF(protectionEndTime,0.0));


        //build Enable data
        ampData.append(QPointF(segmentStartTime,0.0));
        ampData.append(QPointF(twtEnableTime,0.0));
        ampData.append(QPointF(twtEnableTime,1.0));
        ampData.append(QPointF(twtEndTime,1.0));
        ampData.append(QPointF(twtEndTime,0.0));
        ampData.append(QPointF(protectionEndTime,0.0));

    }

    p_chirpCurve->setCurveData(chirpData);
    p_ampEnableCurve->setCurveData(ampData);
    p_protectionCurve->setCurveData(protectionData);

    replot();
}
