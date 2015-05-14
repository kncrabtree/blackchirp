#include "lifdisplaywidget.h"
#include "ui_lifdisplaywidget.h"

#include <QResizeEvent>

#include <qwt6/qwt_matrix_raster_data.h>

LifDisplayWidget::LifDisplayWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::LifDisplayWidget), d_numColumns(-1), d_numRows(-1), d_delayReverse(false),
    d_freqReverse(false)
{
    ui->setupUi(this);

    QFont f(QString("sans-serif"),8);
    QwtText title(QString("Frequency Slice"));
    title.setFont(f);

    ui->freqSlicePlot->setXAxisTitle(QString::fromUtf16(u"Frequency (cm⁻¹)"));
    ui->freqSlicePlot->setTitle(title);
    ui->freqSlicePlot->setName(QString("freqSlicePlot"));

    title.setText(QString("Time Slice"));
    ui->timeSlicePlot->setXAxisTitle(QString::fromUtf16(u"Delay (µs)"));
    ui->timeSlicePlot->setTitle(title);
    ui->timeSlicePlot->setName(QString("timeSlicePlot"));

    title.setText(QString("Current Trace"));
    ui->lifTracePlot->setTitle(title);

    connect(ui->lifTracePlot,&LifTracePlot::colorChanged,this,&LifDisplayWidget::lifColorChanged);

}

LifDisplayWidget::~LifDisplayWidget()
{
    delete ui;
}

void LifDisplayWidget::checkLifColors()
{
    ui->lifTracePlot->checkColors();
}

void LifDisplayWidget::resetLifPlot()
{
    ui->lifTracePlot->reset();
}

void LifDisplayWidget::lifShotAcquired(const LifTrace t)
{
    ui->lifTracePlot->newTrace(t);
}

void LifDisplayWidget::prepareForExperiment(const LifConfig c)
{
    ui->lifTracePlot->clearPlot();
    d_lifData.clear();

    if(!c.isEnabled())
    {
        d_lifData.resize(0);
        d_numColumns = 0;
        d_numRows = 0;

        ui->timeSlicePlot->prepareForExperiment(0.0,1.0);
        ui->freqSlicePlot->prepareForExperiment(0.0,1.0);
    }
    else
    {
        d_numColumns = c.numDelayPoints();
        d_numRows = c.numFrequencyPoints();
        d_lifData.resize(d_numColumns*d_numRows);

        QPair<double,double> delayRange = c.delayRange();
        if(delayRange.first > delayRange.second)
        {
            d_delayReverse = true;
            qSwap(delayRange.first,delayRange.second);
        }
        else
            d_delayReverse = false;

        ui->timeSlicePlot->prepareForExperiment(delayRange.first,delayRange.second);

        QPair<double,double> freqRange = c.frequencyRange();
        if(freqRange.first > freqRange.second)
        {
            d_freqReverse = true;
            qSwap(freqRange.first,freqRange.second);
        }
        else
            d_freqReverse = false;

        ui->freqSlicePlot->prepareForExperiment(freqRange.first,freqRange.second);
    }

    ui->lifSpectrogram->prepareForExperiment(c);
}

void LifDisplayWidget::updatePoint(QPair<QPoint, BlackChirp::LifPoint> val)
{
    QPoint pt = val.first;
    BlackChirp::LifPoint dat = val.second;

    int x = pt.x(), y = pt.y();

    if(d_delayReverse)
        x = d_numColumns - pt.x() - 1;

    if(d_freqReverse)
        y = d_numRows - pt.y() - 1;

    int index = x*d_numRows + y;


    if(index >= 0 && index < d_lifData.size())
    {
        d_lifData[index] = dat;
        ui->lifSpectrogram->updatePoint(x,y,dat.mean);
    }
}

void LifDisplayWidget::resizeEvent(QResizeEvent *ev)
{
    int margin = 5;
    ui->lifTracePlot->setGeometry(0,0,ev->size().width()/3-margin,2*ev->size().height()/5-margin);
    ui->timeSlicePlot->setGeometry(ev->size().width()/3,0,ev->size().width()/3-margin,2*ev->size().height()/5-margin);
    ui->freqSlicePlot->setGeometry(2*ev->size().width()/3,0,ev->size().width()/3-margin,2*ev->size().height()/5-margin);
    ui->lifSpectrogram->setGeometry(0,2*ev->size().height()/5,ev->size().width()-margin,3*ev->size().height()/5-margin);
}
