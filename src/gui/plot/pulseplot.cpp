#include "pulseplot.h"

#include <QApplication>

#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_plot_marker.h>

#include <gui/plot/blackchirpplotcurve.h>
#include <hardware/optional/pulsegenerator/pulsegenerator.h>

PulsePlot::PulsePlot(QWidget *parent) :
    ZoomPanPlot(BC::Key::pulsePlot,parent)
{

    SettingsStorage s(BC::Key::PGen::key,Hardware);
    int numChannels = s.get<int>(BC::Key::PGen::numChannels,8);


    setPlotAxisTitle(QwtPlot::xBottom, QString::fromUtf16(u"Time (μs)"));

    //disable floating for this axis
    axisScaleEngine(QwtPlot::yLeft)->setAttribute(QwtScaleEngine::Floating,false);


    for(int i=0; i<numChannels; i++)
    {
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
        m->setLabel(text);
        m->setLabelAlignment(Qt::AlignLeft);
        m->setValue(0.0, midpoint);
        m->attach(this);
        m->setVisible(false);

        d_plotItems.append({c,m});

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

PulseGenConfig PulsePlot::config()
{
    return d_config;
}

void PulsePlot::newConfig(const PulseGenConfig &c)
{
    d_config = c;
    replot();
}

void PulsePlot::newSetting(int index, PulseGenConfig::Setting s, QVariant val)
{
    if(index < 0 || index > d_config.size())
        return;

    d_config.setCh(index,s,val);
    replot();
}

void PulsePlot::newRepRate(double d)
{
    d_config.d_repRate = d;
}


void PulsePlot::replot()
{
    if(d_config.isEmpty())
    {
        for(int i=0; i<d_plotItems.size();i++)
        {
            d_plotItems.at(i).first->setVisible(false);
            d_plotItems.at(i).second->setVisible(false);
        }

        ZoomPanPlot::replot();
        return;
    }


    double maxTime = 1.0;
    for(int i=0; i<d_config.size(); i++)
    {
        if(d_config.at(i).enabled)
            maxTime = qMax(maxTime,d_config.at(i).delay + d_config.at(i).width);
    }
    maxTime *= 1.25;

    for(int i=0; i<d_config.size() && i <d_plotItems.size(); i++)
    {
        auto c = d_config.at(i);
        double channelOff = (double)(d_config.size()-1-i)*1.5 + 0.25;
        double channelOn = (double)(d_config.size()-1-i)*1.5 + 1.25;
        QVector<QPointF> data;

        if(c.level == PulseGenConfig::ActiveLow)
            qSwap(channelOff,channelOn);

        data.append(QPointF(0.0,channelOff));
        if(c.width > 0.0 && c.enabled)
        {
            data.append(QPointF(c.delay,channelOff));
            data.append(QPointF(c.delay,channelOn));
            data.append(QPointF(c.delay+c.width,channelOn));
            data.append(QPointF(c.delay+c.width,channelOff));
        }
        data.append(QPointF(maxTime,channelOff));

        d_plotItems.at(i).first->setCurveData(data);
        if(!d_plotItems.at(i).first->isVisible())
            d_plotItems.at(i).first->setVisible(true);
        if(c.channelName != d_plotItems.at(i).second->label().text())
        {
            QwtText t = d_plotItems.at(i).second->label();
            t.setText(c.channelName);
            d_plotItems.at(i).second->setLabel(t);
        }
        d_plotItems.at(i).second->setXValue(maxTime);
        if(!d_plotItems.at(i).second->isVisible())
            d_plotItems.at(i).second->setVisible(true);


    }

    ZoomPanPlot::replot();
}

QSize PulsePlot::sizeHint() const
{
    return {500,500};
}
