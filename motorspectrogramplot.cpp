#include "motorspectrogramplot.h"

#include <QMenu>
#include <QMouseEvent>

#include <qwt6/qwt_color_map.h>
#include <qwt6/qwt_scale_widget.h>
#include <qwt6/qwt_interval.h>

MotorSpectrogramPlot::MotorSpectrogramPlot(QWidget *parent) : ZoomPanPlot(QString("motorSpectrogramPlot"),parent)
{
    d_zMax = 1.0;

    setAxisFont(QwtPlot::xBottom,QFont(QString("sans-serif"),8));
    setAxisFont(QwtPlot::yLeft,QFont(QString("sans-serif"),8));

    setAxisAutoScaleRange(QwtPlot::xBottom,0.0,1.0);
    setAxisAutoScaleRange(QwtPlot::yLeft,0.0,1.0);

    connect(this,&MotorSpectrogramPlot::plotRightClicked,this,&MotorSpectrogramPlot::buildContextMenu);

}

void MotorSpectrogramPlot::setLabelText(QwtPlot::Axis axis, QString text)
{
    QwtText label(text);
    label.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(axis,label);
}

void MotorSpectrogramPlot::prepareForScan(const MotorScan s)
{

    if(p_spectrogramData != nullptr)
        delete p_spectrogramData;

    p_spectrogramData = new QwtMatrixRasterData;
    d_firstPoint = true;

    QVector<double> specDat;
    int lPoints = s.numPoints(d_leftAxis);
    int bPoints = s.numPoints(d_bottomAxis);
    specDat.resize(lPoints*bPoints);
    p_spectrogramData->setValueMatrix(specDat,bPoints);

    p_spectrogramData->setInterval(Qt::ZAxis,QwtInterval(0.0,1.0));
    p_spectrogram->setData(p_spectrogramData);
    setAxisAutoScaleRange(QwtPlot::yRight,0.0,1.0);
    autoScale();

    d_zMax = 0.0;
    replot();
}

void MotorSpectrogramPlot::buildContextMenu(QMouseEvent *me)
{
    QMenu *m = contextMenu();

    m->popup(me->globalPos());

}

void MotorSpectrogramPlot::updateData(QVector<double> data, int cols, double max, MotorScan::MotorDataAxis leftAxis, MotorScan::MotorDataAxis bottomAxis)
{
    if(leftAxis == d_leftAxis && bottomAxis == d_bottomAxis)
    {
        if(max > d_zMax)
        {
            setAxisAutoScaleRange(QwtPlot::yRight,0.0,max);
            p_spectrogramData->setInterval(Qt::ZAxis,QwtInterval(0.0,max));

            QwtLinearColorMap *map2 = new QwtLinearColorMap(Qt::black,Qt::red);
            map2->addColorStop(0.05,Qt::darkBlue);
            map2->addColorStop(0.1,Qt::blue);
            map2->addColorStop(0.25,Qt::cyan);
            map2->addColorStop(0.5,Qt::green);
            map2->addColorStop(0.75,Qt::yellow);
            map2->addColorStop(0.9,QColor(0xff,0x66,0));

            QwtScaleWidget *rightAxis = axisWidget( QwtPlot::yRight );
            rightAxis->setColorMap(QwtInterval(0.0,max),map2);

            d_zMax = max;
        }

        p_spectrogramData->setValueMatrix(data,cols);
    }

    replot();
}

void MotorSpectrogramPlot::setAxis(QwtPlot::Axis plotAxis, MotorScan::MotorDataAxis motorAxis)
{
    QString text;

    switch (motorAxis) {
    case MotorScan::MotorX:
        text = QString("X (mm)");
        break;
    case MotorScan::MotorY:
        text = QString("Y (mm)");
        break;
    case MotorScan::MotorZ:
        text = QString("Z (mm)");
        break;
    case MotorScan::MotorT:
        text = QString::fromUtf16(u"T (µs)");
        break;
    }

    if(plotAxis == QwtPlot::yLeft || plotAxis == QwtPlot::yRight)
        d_leftAxis = motorAxis;
    else
        d_bottomAxis = motorAxis;

    setLabelText(plotAxis,text);
}



void MotorSpectrogramPlot::filterData()
{
}
