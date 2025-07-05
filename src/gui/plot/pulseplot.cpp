#include "pulseplot.h"
#include <gui/plot/curvefactory.h>

#include <QApplication>

#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_plot_marker.h>

#include <gui/plot/blackchirpplotcurve.h>
#include <hardware/optional/pulsegenerator/pulsegenerator.h>

PulsePlot::PulsePlot(std::shared_ptr<PulseGenConfig> cfg, QWidget *parent) :
    ZoomPanPlot(BC::Key::pulsePlot,parent), ps_config{cfg}
{

    int numChannels = 0;
    if(auto p = ps_config.lock())
        numChannels = p->d_channels.size();

    setPlotAxisTitle(QwtPlot::xBottom, QString::fromUtf16(u"Time (μs)"));

    //disable floating for this axis
    axisScaleEngine(QwtPlot::yLeft)->setAttribute(QwtScaleEngine::Floating,false);

    // Disable QwtPlot's automatic memory management
    setAutoDelete(false);

    for(int i=0; i<numChannels; i++)
    {
        double bottom = (double)(numChannels - 1 - i)*1.5;
        double midpoint = (double)(numChannels - 1 - i)*1.5 + 0.75;
        double top = (double)(numChannels-i)*1.5;

        auto c = CurveFactory::createStandardCurve<BlackchirpPlotCurve>(BC::Key::pulseChannel+QString::number(i));
        c->attach(this);
        c->setVisible(false);

        auto p = c->pen();
        p.setWidth(0);
        p.setStyle(Qt::DotLine);

        auto sep = std::make_unique<QwtPlotMarker>();
        sep->setLineStyle(QwtPlotMarker::HLine);
        sep->setLinePen(p);
        sep->setYValue(top);
        sep->attach(this);
        sep->setItemAttribute(QwtPlotItem::AutoScale);

        p.setStyle(Qt::NoPen);
        auto sep2 = std::make_unique<QwtPlotMarker>();
        sep2->setLineStyle(QwtPlotMarker::HLine);
        sep2->setLinePen(p);
        sep2->setYValue(top);
        sep2->setYAxis(QwtPlot::yRight);
        sep2->attach(this);
        sep2->setItemAttribute(QwtPlotItem::AutoScale);

        auto m = std::make_unique<QwtPlotMarker>();
        QwtText text;
        text.setColor(p.color());
        text.setRenderFlags(Qt::AlignRight | Qt::AlignVCenter);
        text.setBackgroundBrush(QBrush(QPalette().color(QPalette::Window)));
        QColor border = QPalette().color(QPalette::Text);
        border.setAlpha(0);
        text.setBorderPen(QPen(border));
        text.setColor(QPalette().color(QPalette::Text));
        m->setLabel(text);

        m->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);

        m->setValue(0.0, midpoint);
        m->attach(this);
        m->setVisible(false);

        auto sync = std::make_unique<QwtPlotCurve>();
        QPalette pal;
        p.setColor(pal.color(QPalette::Text));
        p.setWidthF(1.0);
        p.setStyle(Qt::DashLine);
        sync->setPen(p);
        sync->setVisible(false);
        sync->attach(this);

        PlotItem item;
        item.min = bottom;
        item.max = top;
        item.mid = midpoint;
        item.curve = std::move(c);
        item.labelMarker = std::move(m);
        item.syncCurve = std::move(sync);
        item.separator = std::move(sep);
        item.separator2 = std::move(sep2);
        d_plotItems.push_back(std::move(item));

    }

    setAxisOverride(QwtPlot::yLeft);
    enableAxis(QwtPlot::yLeft,false);
    setAxisOverride(QwtPlot::yRight);
    enableAxis(QwtPlot::yRight,false);

    replot();
}

PulsePlot::~PulsePlot()
{
    // All items are managed by unique_ptr and will be automatically cleaned up
}

void PulsePlot::updatePulsePlot()
{
    replot();
}

void PulsePlot::newConfig(std::shared_ptr<PulseGenConfig> c)
{
    ps_config = c;
    replot();
}


void PulsePlot::replot()
{
    auto c = ps_config.lock();
    if(!c)
    {
        for(auto &item : d_plotItems)
        {
            item.curve->setVisible(false);
            item.labelMarker->setVisible(false);
            item.syncCurve->setVisible(false);
        }

        ZoomPanPlot::replot();
        return;
    }


    double maxTime = 1.0;
    for(int i=0; i<c->size(); i++)
    {
        if(c->at(i).enabled)
            maxTime = qMax(maxTime,c->channelStart(i) + c->at(i).width);
    }
    maxTime *= 1.25;

    auto cit = c->d_channels.cbegin();
    int plotIndex = 0;

    for(int i=0 ;cit != c->d_channels.cend() && plotIndex < d_plotItems.size(); ++cit, ++plotIndex, ++i)
    {
        auto &plotItem = d_plotItems[plotIndex];
        double channelOff = plotItem.min + 0.25;
        double channelOn = plotItem.max - 0.25;
        QVector<QPointF> data;

        if(cit->level == PulseGenConfig::ActiveLow)
            qSwap(channelOff,channelOn);

        if(cit->syncCh > 0 && cit->enabled)
        {
            auto offset = c->channelStart(cit->syncCh-1);
            plotItem.syncCurve->setSamples({{offset,plotItem.min},{offset,plotItem.max}});
            plotItem.syncCurve->setVisible(true);
        }
        else
            plotItem.syncCurve->setVisible(false);

        data.append(QPointF(0.0,channelOff));
        if(cit->width > 0.0 && cit->enabled)
        {
            data.append(QPointF(c->channelStart(i),channelOff));
            data.append(QPointF(c->channelStart(i),channelOn));
            data.append(QPointF(c->channelStart(i)+cit->width,channelOn));
            data.append(QPointF(c->channelStart(i)+cit->width,channelOff));
        }
        data.append(QPointF(maxTime,channelOff));

        plotItem.curve->setCurveData(data);
        plotItem.curve->setVisible(true);

        QString label = cit->channelName;
        if(label.isEmpty())
            label = QString("Ch%1").arg(i+1);
        if(cit->mode == PulseGenConfig::Normal)
        {
            if(c->d_mode == PulseGenConfig::Continuous)
                label.append(QString("\n%1 Hz").arg(c->d_repRate,0,'f',2));
        }
        else
        {
            if(cit->dutyOn == 1)
                label.append(QString("\nDUTY: %1 Hz").arg(c->d_repRate/(cit->dutyOff+1),0,'f',2));
            else
                label.append(QString("\nDUTY: %1 On/%2 Off").arg(cit->dutyOn).arg(cit->dutyOff));
        }
        QwtText t = plotItem.labelMarker->label();
        t.setText(label);
        plotItem.labelMarker->setLabel(t);
        plotItem.labelMarker->setXValue(maxTime);
        plotItem.labelMarker->setVisible(true);


    }

    ZoomPanPlot::replot();
}

QSize PulsePlot::sizeHint() const
{
    return {500,500};
}
