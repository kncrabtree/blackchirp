#include "lifdisplaywidget.h"
#include "ui_lifdisplaywidget.h"

#include <QResizeEvent>
#include <math.h>

#include <qwt6/qwt_matrix_raster_data.h>

LifDisplayWidget::LifDisplayWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::LifDisplayWidget), d_numFrequencyPoints(-1), d_numDelayPoints(-1), d_delayReverse(false),
    d_freqReverse(false), d_currentSpectrumDelayIndex(-1), d_currentTimeTraceFreqIndex(-1)
{
    ui->setupUi(this);

    ui->spectrumPlot->setXAxisTitle(QString::fromUtf16(u"Frequency (cm⁻¹)"));
    ui->spectrumPlot->setPlotTitle(QString("Frequency Slice"));
    ui->spectrumPlot->setName(QString("spectrumPlot"));

    ui->timeTracePlot->setXAxisTitle(QString::fromUtf16(u"Delay (µs)"));
    ui->timeTracePlot->setPlotTitle(QString("Time Slice"));
    ui->timeTracePlot->setName(QString("timeTracePlot"));

    QwtText title;
    title.setText(QString("Current Trace"));
    title.setFont(QFont("sans-serif",8));
    ui->lifTracePlot->setTitle(title);
    ui->lifTracePlot->setDisplayOnly(true);

    connect(ui->lifTracePlot,&LifTracePlot::colorChanged,this,&LifDisplayWidget::lifColorChanged);
    connect(ui->lifSpectrogram,&LifSpectrogramPlot::freqSlice,this,&LifDisplayWidget::freqSlice);
    connect(ui->lifSpectrogram,&LifSpectrogramPlot::delaySlice,this,&LifDisplayWidget::delaySlice);

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

    d_currentTimeTraceFreqIndex = -1;
    d_currentSpectrumDelayIndex = -1;

    ui->spectrumPlot->setPlotTitle(QString("Frequency Slice"));
    ui->timeTracePlot->setPlotTitle(QString("Time Slice"));


    if(!c.isEnabled())
    {
        d_lifData.resize(0);
        d_numFrequencyPoints = 0;
        d_numDelayPoints = 0;

        ui->timeTracePlot->prepareForExperiment(0.0,1.0);
        ui->spectrumPlot->prepareForExperiment(0.0,1.0);
    }
    else
    {
        d_numDelayPoints = c.numDelayPoints();
        d_numFrequencyPoints = c.numFrequencyPoints();
        d_lifData.resize(d_numFrequencyPoints*d_numDelayPoints);

        d_delayRange = c.delayRange();
        if(d_delayRange.first > d_delayRange.second)
        {
            d_delayReverse = true;
            qSwap(d_delayRange.first,d_delayRange.second);
        }
        else
            d_delayReverse = false;

        ui->timeTracePlot->prepareForExperiment(d_delayRange.first,d_delayRange.second);

        d_freqRange = c.frequencyRange();
        if(d_freqRange.first > d_freqRange.second)
        {
            d_freqReverse = true;
            qSwap(d_freqRange.first,d_freqRange.second);
        }
        else
            d_freqReverse = false;

        ui->spectrumPlot->prepareForExperiment(d_freqRange.first,d_freqRange.second);

    }

    ui->lifSpectrogram->prepareForExperiment(c);
}

void LifDisplayWidget::updatePoint(QPair<QPoint, BlackChirp::LifPoint> val)
{
    //x is delay index, y is frequency index
    QPoint pt = val.first;
    BlackChirp::LifPoint dat = val.second;

    int delayIndex = pt.x(), freqIndex = pt.y();

    if(d_delayReverse)
        delayIndex = d_numDelayPoints - pt.x() - 1;

    if(d_freqReverse)
        freqIndex = d_numFrequencyPoints - pt.y() - 1;

    int index = delayIndex*d_numFrequencyPoints + freqIndex;


    if(index >= 0 && index < d_lifData.size())
    {
        d_lifData[index] = dat;
        ui->lifSpectrogram->updatePoint(delayIndex,freqIndex,dat.mean);
    }

    if(freqIndex == d_currentTimeTraceFreqIndex)
        updateTimeTrace();

    if(delayIndex == d_currentSpectrumDelayIndex)
        updateSpectrum();
}

void LifDisplayWidget::freqSlice(int delayIndex)
{
    if(delayIndex == d_currentSpectrumDelayIndex)
        return;

    d_currentSpectrumDelayIndex = delayIndex;
    updateSpectrum();
}

void LifDisplayWidget::delaySlice(int freqIndex)
{
    if(freqIndex == d_currentTimeTraceFreqIndex)
        return;

    d_currentTimeTraceFreqIndex = freqIndex;
    updateTimeTrace();
}

void LifDisplayWidget::updateSpectrum()
{
    QVector<QPointF> slice;
    slice.resize(d_numFrequencyPoints);
    double delta = fabs(d_freqRange.second-d_freqRange.first)/static_cast<double>(d_numFrequencyPoints-1);
    double max = 0.0;
    for(int i=0; i < d_numFrequencyPoints; i++)
    {
        double dat = d_lifData.at(d_currentSpectrumDelayIndex*d_numFrequencyPoints + i).mean;
        max = qMax(max,dat);
        slice[i] = QPointF(d_freqRange.first + static_cast<double>(i)*delta,dat);
    }

    delta = fabs(d_delayRange.second-d_delayRange.first)/static_cast<double>(d_numDelayPoints-1);
    double dVal = d_delayRange.first + static_cast<double>(d_currentSpectrumDelayIndex)*delta;
    QString labelText = QString::fromUtf16(u"Spectrum at %1 µs").arg(dVal,0,'f',2);
    ui->spectrumPlot->setPlotTitle(labelText);
    ui->spectrumPlot->setAxisAutoScaleRange(QwtPlot::yLeft,0.0,max);
    ui->spectrumPlot->setData(slice);
}

void LifDisplayWidget::updateTimeTrace()
{
    QVector<QPointF> slice;
    slice.resize(d_numDelayPoints);
    double delta = fabs(d_delayRange.second-d_delayRange.first)/static_cast<double>(d_numDelayPoints-1);
    double max = 0.0;
    for(int i=0; i < d_numDelayPoints; i++)
    {
        double dat = d_lifData.at(i*d_numFrequencyPoints + d_currentTimeTraceFreqIndex).mean;
        slice[i] = QPointF(d_delayRange.first + static_cast<double>(i)*delta,dat);
        max = qMax(max,dat);
    }

    ui->timeTracePlot->setAxisAutoScaleRange(QwtPlot::yLeft,0.0,max);

    delta = fabs(d_freqRange.second-d_freqRange.first)/static_cast<double>(d_numFrequencyPoints-1);
    double fVal = d_freqRange.first + static_cast<double>(d_currentTimeTraceFreqIndex)*delta;
    QString labelText = QString::fromUtf16(u"Time Slice at %1 cm⁻¹").arg(fVal,0,'f',3);
    ui->timeTracePlot->setPlotTitle(labelText);
    ui->timeTracePlot->setData(slice);
}

void LifDisplayWidget::resizeEvent(QResizeEvent *ev)
{
    int margin = 5;
    ui->lifTracePlot->setGeometry(0,0,ev->size().width()/3-margin,2*ev->size().height()/5-margin);
    ui->timeTracePlot->setGeometry(ev->size().width()/3,0,ev->size().width()/3-margin,2*ev->size().height()/5-margin);
    ui->spectrumPlot->setGeometry(2*ev->size().width()/3,0,ev->size().width()/3-margin,2*ev->size().height()/5-margin);
    ui->lifSpectrogram->setGeometry(0,2*ev->size().height()/5,ev->size().width()-margin,3*ev->size().height()/5-margin);
}
