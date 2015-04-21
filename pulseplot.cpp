#include "pulseplot.h"

#include <QSettings>
#include <QApplication>

#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_plot_marker.h>

#include "pulsegenerator.h"

PulsePlot::PulsePlot(QWidget *parent) :
    ZoomPanPlot(QString("pulsePlot"),parent)
{

    int numChannels = BC_PGEN_NUMCHANNELS;

    setTitle(QwtText("Pulses"));

    QFont labelFont(QString("sans serif"),8);
    QwtText xLabel(QString("Time (<span>&mu;</span>s)"));
    xLabel.setFont(labelFont);
    setAxisTitle(QwtPlot::xBottom, xLabel);
    enableAxis(QwtPlot::yLeft,false);


    setAxisAutoScaleRange(QwtPlot::yLeft,0.0,numChannels*1.5);

    QPen p(QPalette().color(QPalette::Text));
    QPen dotP(p);
    dotP.setStyle(Qt::DotLine);
    for(int i=0; i<numChannels; i++)
    {
        double midpoint = (double)(numChannels - 1 - i)*1.5 + 0.75;
        double top = (double)(numChannels-i)*1.5;

        QwtPlotMarker *sep = new QwtPlotMarker;
        sep->setLineStyle(QwtPlotMarker::HLine);
        sep->setLinePen(dotP);
        sep->setYValue(top);
        sep->attach(this);

        QwtPlotCurve *c = new QwtPlotCurve;
        c->setPen(p);
        c->attach(this);
        c->setVisible(false);

        QwtPlotMarker *m = new QwtPlotMarker;
        QwtText text;
        text.setFont(labelFont);
        text.setColor(QPalette().color(QPalette::Text));
        m->setLabel(text);
        m->setLabelAlignment(Qt::AlignLeft);
        m->setValue(0.0, midpoint);
        m->attach(this);
        m->setVisible(false);

        d_plotItems.append(QPair<QwtPlotCurve*,QwtPlotMarker*>(c,m));

    }

    replot();
}

PulsePlot::~PulsePlot()
{

}

PulseGenConfig PulsePlot::config()
{
    return d_config;
}

void PulsePlot::newConfig(const PulseGenConfig c)
{
    d_config = c;
    replot();
}

void PulsePlot::newSetting(int index, PulseGenConfig::Setting s, QVariant val)
{
    if(index < 0 || index > d_config.size())
        return;

    d_config.set(index,s,val);
    replot();
}

void PulsePlot::newRepRate(double d)
{
    d_config.setRepRate(d);
}



void PulsePlot::filterData()
{
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
        setAxisAutoScaleRange(QwtPlot::xBottom,0.0,1.0);
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
    setAxisAutoScaleRange(QwtPlot::xBottom,0.0,maxTime);

    for(int i=0; i<d_config.size() && i <d_plotItems.size(); i++)
    {
        PulseGenConfig::ChannelConfig c = d_config.at(i);
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

        d_plotItems.at(i).first->setSamples(data);
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
