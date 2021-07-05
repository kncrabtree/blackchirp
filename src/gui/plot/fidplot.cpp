#include <gui/plot/fidplot.h>
#include <math.h>

#include <QPalette>
#include <QApplication>
#include <QMenu>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QWidgetAction>
#include <QFormLayout>
#include <QColorDialog>
#include <QMouseEvent>
#include <QFileDialog>
#include <QMessageBox>

#include <qwt6/qwt_plot_canvas.h>
#include <qwt6/qwt_plot_marker.h>
#include <gui/plot/blackchirpplotcurve.h>

FidPlot::FidPlot(const QString id, QWidget *parent) :
    ZoomPanPlot(BC::Key::fidPlot+id,parent)
{

    setPlotAxisTitle(QwtPlot::xBottom,QString::fromUtf16(u"Time (μs)"));
    setPlotAxisTitle(QwtPlot::yLeft,QString("FID ")+id);

    p_curve = new BlackchirpPlotCurve(QString("FID")+id);
    p_curve->attach(this);

    QwtPlotMarker *chirpStartMarker = new QwtPlotMarker();
    chirpStartMarker->setLineStyle(QwtPlotMarker::VLine);
    chirpStartMarker->setLinePen(QPen(QPalette().color(QPalette::BrightText)));
    QwtText csLabel(QString("Chirp Start"));
    csLabel.setFont(QFont(QString("sans serif"),6));
    chirpStartMarker->setLabel(csLabel);
    chirpStartMarker->setLabelOrientation(Qt::Vertical);
    chirpStartMarker->setLabelAlignment(Qt::AlignBottom|Qt::AlignRight);
    d_chirpMarkers.first = chirpStartMarker;
    chirpStartMarker->attach(this);
    chirpStartMarker->setVisible(false);

    QwtPlotMarker *chirpEndMarker = new QwtPlotMarker();
    chirpEndMarker->setLineStyle(QwtPlotMarker::VLine);
    chirpEndMarker->setLinePen(QPen(QPalette().color(QPalette::Text)));
    QwtText ceLabel(QString("Chirp End"));
    ceLabel.setFont(QFont(QString("sans serif"),6));
    chirpEndMarker->setLabel(ceLabel);
    chirpEndMarker->setLabelOrientation(Qt::Vertical);
    chirpEndMarker->setLabelAlignment(Qt::AlignTop|Qt::AlignRight);
    d_chirpMarkers.second = chirpEndMarker;
    chirpEndMarker->attach(this);
    chirpEndMarker->setVisible(false);


    QwtPlotMarker *ftStartMarker = new QwtPlotMarker();
    ftStartMarker->setLineStyle(QwtPlotMarker::VLine);
    ftStartMarker->setLinePen(QPen(QPalette().color(QPalette::Text)));
    QwtText ftsLabel(QString(" FT Start "));
    ftsLabel.setFont(QFont(QString("sans serif"),6));
    ftsLabel.setBackgroundBrush(QPalette().window());
    ftsLabel.setColor(QPalette().text().color());
    ftStartMarker->setLabel(ftsLabel);
    ftStartMarker->setLabelOrientation(Qt::Vertical);
    ftStartMarker->setLabelAlignment(Qt::AlignBottom|Qt::AlignRight);
    ftStartMarker->setXValue(0.0);
    ftStartMarker->attach(this);
    ftStartMarker->setVisible(false);
    d_ftMarkers.first = ftStartMarker;

    QwtPlotMarker *ftEndMarker = new QwtPlotMarker();
    ftEndMarker->setLineStyle(QwtPlotMarker::VLine);
    ftEndMarker->setLinePen(QPen(QPalette().color(QPalette::Text)));
    QwtText fteLabel(QString(" FT End "));
    fteLabel.setFont(QFont(QString("sans serif"),6));
    fteLabel.setBackgroundBrush(QPalette().window());
    fteLabel.setColor(QPalette().text().color());
    ftEndMarker->setLabel(fteLabel);
    ftEndMarker->setLabelOrientation(Qt::Vertical);
    ftEndMarker->setLabelAlignment(Qt::AlignTop|Qt::AlignLeft);
    ftEndMarker->setXValue(0.0);
    ftEndMarker->attach(this);
    ftEndMarker->setVisible(false);
    d_ftMarkers.second = ftEndMarker;

}

void FidPlot::receiveProcessedFid(const QVector<QPointF> d)
{
    p_curve->setCurveData(d);

    replot();
}

void FidPlot::prepareForExperiment(const Experiment e)
{     
    auto &c = e.d_ftmwCfg;
    p_curve->setCurveData(QVector<QPointF>());

    if(!c.isEnabled())
    {
        p_curve->setVisible(false);

        d_chirpMarkers.first->setVisible(false);
        d_chirpMarkers.second->setVisible(false);
        d_ftMarkers.first->setVisible(false);
        d_ftMarkers.second->setVisible(false);
    }
    else
    {
        p_curve->setVisible(true);

        d_ftMarkers.first->setVisible(true);
        d_ftMarkers.second->setVisible(true);

        double maxTime = (static_cast<double>(c.scopeConfig().d_recordLength)-1.0)/c.scopeConfig().d_sampleRate*1e6;
        double ftEnd = d_ftMarkers.second->xValue();
        if(ftEnd <= 0.0 || ftEnd <= d_ftMarkers.first->xValue() || ftEnd > maxTime)
            d_ftMarkers.second->setXValue(maxTime);

        emit ftStartChanged(d_ftMarkers.first->xValue());
        emit ftEndChanged(d_ftMarkers.second->xValue());

        bool displayMarkers = c.isPhaseCorrectionEnabled() || c.isChirpScoringEnabled();
        if(displayMarkers)
        {
            ///TODO: Update this calculation!
            double chirpStart = c.d_rfConfig.getChirpConfig().preChirpGateDelay() + c.d_rfConfig.getChirpConfig().preChirpProtectionDelay() - c.scopeConfig().d_triggerDelayUSec;
            double chirpEnd = chirpStart + c.d_rfConfig.getChirpConfig().chirpDuration(0);

            d_chirpMarkers.first->setValue(chirpStart,0.0);
            d_chirpMarkers.second->setValue(chirpEnd,0.0);
        }

        d_chirpMarkers.first->setVisible(displayMarkers);
        d_chirpMarkers.second->setVisible(displayMarkers);
    }

    autoScale();
}

void FidPlot::setFtStart(double start)
{
    double v = start;
    auto currentFid = p_curve->curveData();
    if(!currentFid.isEmpty())
        v = qBound(0.0,start,qMin(d_ftMarkers.second->value().x(),currentFid.constLast().x()*1e6));

    d_ftMarkers.first->setValue(v,0.0);

    emit ftStartChanged(v);

    replot();
}

void FidPlot::setFtEnd(double end)
{
    double v = end;
    auto currentFid = p_curve->curveData();
    if(!currentFid.isEmpty())
        v = qBound(d_ftMarkers.first->value().x(),end,currentFid.constLast().x()*1e6);

    d_ftMarkers.second->setValue(v,0.0);
    emit ftEndChanged(v);

    replot();
}
