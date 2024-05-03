#include "pulseplot.h"

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


    for(int i=0; i<numChannels; i++)
    {
        double bottom = (double)(numChannels - 1 - i)*1.5;
        double midpoint = (double)(numChannels - 1 - i)*1.5 + 0.75;
        double top = (double)(numChannels-i)*1.5;

        BlackchirpPlotCurve *c = new BlackchirpPlotCurve(BC::Key::pulseChannel+QString::number(i));
        c->attach(this);
        c->setVisible(false);

        auto p = c->pen();
        p.setWidth(0);
        p.setStyle(Qt::DotLine);

        QwtPlotMarker *sep = new QwtPlotMarker;
        sep->setLineStyle(QwtPlotMarker::HLine);
        sep->setLinePen(p);
        sep->setYValue(top);
        sep->attach(this);
        sep->setItemAttribute(QwtPlotItem::AutoScale);

        p.setStyle(Qt::NoPen);
        QwtPlotMarker *sep2 = new QwtPlotMarker;
        sep2->setLineStyle(QwtPlotMarker::HLine);
        sep2->setLinePen(p);
        sep2->setYValue(top);
        sep2->setYAxis(QwtPlot::yRight);
        sep2->attach(this);
        sep2->setItemAttribute(QwtPlotItem::AutoScale);



        QwtPlotMarker *m = new QwtPlotMarker;
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

        QwtPlotCurve *sync = new QwtPlotCurve;
        QPalette pal;
        p.setColor(pal.color(QPalette::Text));
        p.setWidthF(1.0);
        p.setStyle(Qt::DashLine);
        sync->setPen(p);
        sync->setVisible(false);
        sync->attach(this);


        d_plotItems.append({bottom,top,midpoint,c,m,sync});

    }

    setAxisOverride(QwtPlot::yLeft);
    enableAxis(QwtPlot::yLeft,false);
    setAxisOverride(QwtPlot::yRight);
    enableAxis(QwtPlot::yRight,false);

    replot();
}

PulsePlot::~PulsePlot()
{

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
        for(auto it = d_plotItems.begin(); it != d_plotItems.end(); ++it)
        {
            it->curve->setVisible(false);
            it->labelMarker->setVisible(false);
            it->syncCurve->setVisible(false);
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
    auto pit = d_plotItems.begin();

    for(int i=0 ;cit != c->d_channels.cend() && pit != d_plotItems.end(); ++cit, ++pit, ++i)
    {
        double channelOff = pit->min + 0.25;
        double channelOn = pit->max - 0.25;
        QVector<QPointF> data;

        if(cit->level == PulseGenConfig::ActiveLow)
            qSwap(channelOff,channelOn);

        if(cit->syncCh > 0 && cit->enabled)
        {
            auto offset = c->channelStart(cit->syncCh-1);
            pit->syncCurve->setSamples({{offset,pit->min},{offset,pit->max}});
            pit->syncCurve->setVisible(true);
        }
        else
            pit->syncCurve->setVisible(false);

        data.append(QPointF(0.0,channelOff));
        if(cit->width > 0.0 && cit->enabled)
        {
            data.append(QPointF(c->channelStart(i),channelOff));
            data.append(QPointF(c->channelStart(i),channelOn));
            data.append(QPointF(c->channelStart(i)+cit->width,channelOn));
            data.append(QPointF(c->channelStart(i)+cit->width,channelOff));
        }
        data.append(QPointF(maxTime,channelOff));

        pit->curve->setCurveData(data);
        pit->curve->setVisible(true);

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
        QwtText t = pit->labelMarker->label();
        t.setText(label);
        pit->labelMarker->setLabel(t);
        pit->labelMarker->setXValue(maxTime);
        pit->labelMarker->setVisible(true);


    }

    ZoomPanPlot::replot();
}

QSize PulsePlot::sizeHint() const
{
    return {500,500};
}
