#include <src/modules/lif/gui/lifdisplaywidget.h>
#include "ui_lifdisplaywidget.h"

#include <QResizeEvent>
#include <math.h>

#include <qwt6/qwt_matrix_raster_data.h>

LifDisplayWidget::LifDisplayWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::LifDisplayWidget), d_delayReverse(false), d_freqReverse(false), d_currentSpectrumDelayIndex(-1), d_currentTimeTraceFreqIndex(-1)
{
    ui->setupUi(this);

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("lifLaser"));
    s.beginGroup(s.value(QString("subKey"),QString("virtual")).toString());
    d_laserUnits = s.value(QString("units"),QString("nm")).toString();
    s.endGroup();
    s.endGroup();

    ui->spectrumPlot->setXAxisTitle(QString("Laser Position (%1)").arg(d_laserUnits));
    ui->spectrumPlot->setPlotTitle(QString("Laser Slice"));
    ui->spectrumPlot->setName(QString("spectrumPlot"));

    ui->timeTracePlot->setXAxisTitle(QString::fromUtf16(u"Delay (µs)"));
    ui->timeTracePlot->setPlotTitle(QString("Time Slice"));
    ui->timeTracePlot->setName(QString("timeTracePlot"));

    QwtText title;
    title.setText(QString("Time Trace"));
    title.setFont(QFont("sans-serif",8));
    ui->lifTracePlot->setTitle(title);
    ui->lifTracePlot->setDisplayOnly(true);

    connect(ui->lifTracePlot,&LifTracePlot::colorChanged,this,&LifDisplayWidget::lifColorChanged);
    connect(ui->lifTracePlot,&LifTracePlot::lifGateUpdated,this,&LifDisplayWidget::lifZoneUpdate);
    connect(ui->lifTracePlot,&LifTracePlot::refGateUpdated,this,&LifDisplayWidget::refZoneUpdate);
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

void LifDisplayWidget::prepareForExperiment(const LifConfig c)
{
    ui->lifTracePlot->clearPlot();




    ui->spectrumPlot->setPlotTitle(QString("Laser Slice"));
    ui->timeTracePlot->setPlotTitle(QString("Time Slice"));


    if(!c.isEnabled())
    {
        ui->timeTracePlot->prepareForExperiment(0.0,1.0);
        ui->spectrumPlot->prepareForExperiment(0.0,1.0);
        d_currentTimeTraceFreqIndex = -1;
        d_currentSpectrumDelayIndex = -1;
    }
    else
    {

        if(c.numDelayPoints() > 1)
            ui->timeTracePlot->prepareForExperiment(qMin(c.delayRange().first,c.delayRange().second),qMax(c.delayRange().first,c.delayRange().second));
        else
            ui->timeTracePlot->prepareForExperiment(qMin(c.delayRange().first + c.delayStep(),c.delayRange().first - c.delayStep()),
                                                    qMax(c.delayRange().first + c.delayStep(),c.delayRange().first - c.delayStep()));
        if(c.numLaserPoints() > 1)
            ui->spectrumPlot->prepareForExperiment(qMin(c.laserRange().first,c.laserRange().second),qMax(c.laserRange().first,c.laserRange().second));
        else
            ui->spectrumPlot->prepareForExperiment(qMin(c.laserRange().first + c.laserStep(),c.laserRange().first - c.laserStep()),
                                                    qMax(c.laserRange().first + c.laserStep(),c.laserRange().first - c.laserStep()));

        d_currentTimeTraceFreqIndex = 0;
        d_currentSpectrumDelayIndex = 0;
    }

    ui->lifSpectrogram->prepareForExperiment(c);
    d_currentLifConfig = c;
}

void LifDisplayWidget::updatePoint(const LifConfig c)
{
    //don't overwrite integration ranges if user has changed them
    auto lr = d_currentLifConfig.lifGate();
    auto rr = d_currentLifConfig.refGate();
    d_currentLifConfig = c;
    d_currentLifConfig.setLifGate(lr);
    if(d_currentLifConfig.scopeConfig().refEnabled)
        d_currentLifConfig.setRefGate(rr);

    auto zRange = integrate(d_currentLifConfig);
    ui->lifSpectrogram->updateData(d_currentIntegratedData,d_currentLifConfig.numLaserPoints(),zRange.first,zRange.second);


    updateTimeTrace();
    updateSpectrum();
    updateLifTrace();
}

void LifDisplayWidget::freqSlice(int delayIndex)
{
    if(d_currentLifConfig.delayStep() > 0.0)
        d_currentSpectrumDelayIndex = delayIndex;
    else
        d_currentSpectrumDelayIndex = d_currentLifConfig.numDelayPoints()-1-delayIndex;

    updateSpectrum();
    updateLifTrace();
}

void LifDisplayWidget::delaySlice(int freqIndex)
{
    if(d_currentLifConfig.laserStep() > 0.0)
        d_currentTimeTraceFreqIndex = freqIndex;
    else
        d_currentTimeTraceFreqIndex = d_currentLifConfig.numLaserPoints()-1-freqIndex;

    updateTimeTrace();
    updateLifTrace();
}

void LifDisplayWidget::lifZoneUpdate(int min, int max)
{
    d_currentLifConfig.setLifGate(min,max);
    updatePoint(d_currentLifConfig);

}

void LifDisplayWidget::refZoneUpdate(int min, int max)
{
    d_currentLifConfig.setRefGate(min,max);
    updatePoint(d_currentLifConfig);
}

void LifDisplayWidget::updateSpectrum()
{
    int nlp = d_currentLifConfig.numLaserPoints();
    QVector<QPointF> slice(nlp);

    double min=0.0, max = 0.0;
    int offset = 0;

    //sort from lowest laser position to highest
    if(d_currentLifConfig.laserStep() < 0.0)
        offset = nlp-1;

    int j = d_currentSpectrumDelayIndex;
    if(d_currentLifConfig.delayStep() < 0.0)
        j = d_currentLifConfig.numDelayPoints()-j-1;

    for(int i=0; i < nlp; i++)
    {

        double dat = d_currentIntegratedData.at(j*nlp + qAbs(i-offset));
        max = qMax(max,dat);
        min = qMin(min,dat);
        slice[qAbs(i-offset)].setX(d_currentLifConfig.laserRange().first + i*d_currentLifConfig.laserStep());
        slice[qAbs(i-offset)].setY(dat);
    }

    double dVal = d_currentLifConfig.delayRange().first + d_currentSpectrumDelayIndex*d_currentLifConfig.delayStep();
    QString labelText = QString::fromUtf16(u"Spectrum at %1 µs").arg(dVal,0,'f',2);
    ui->spectrumPlot->setPlotTitle(labelText);
    ui->spectrumPlot->setAxisAutoScaleRange(QwtPlot::yLeft,min,max);
    ui->spectrumPlot->setData(slice);
}

void LifDisplayWidget::updateTimeTrace()
{
    int ndp = d_currentLifConfig.numDelayPoints();
    int nlp = d_currentLifConfig.numLaserPoints();
    QVector<QPointF> slice(ndp);

    int offset = 0;
    if(d_currentLifConfig.delayStep()<0.0)
        offset = ndp-1;

    double min=0.0, max = 0.0;
    int j = d_currentTimeTraceFreqIndex;
    if(d_currentLifConfig.laserStep() < 0.0)
        j = d_currentLifConfig.numLaserPoints()-j-1;

    for(int i=0; i < ndp; i++)
    {

        double dat = d_currentIntegratedData.at(qAbs(i-offset)*nlp+ j);
        slice[qAbs(i-offset)].setX(d_currentLifConfig.delayRange().first + i*d_currentLifConfig.delayStep());
        slice[qAbs(i-offset)].setY(dat);
        max = qMax(max,dat);
        min = qMin(min,dat);
    }

    ui->timeTracePlot->setAxisAutoScaleRange(QwtPlot::yLeft,min,max);

    double fVal = d_currentLifConfig.laserRange().first + d_currentTimeTraceFreqIndex*d_currentLifConfig.laserStep();
    QString labelText = QString::fromUtf16(u"Time Slice at %1 %2").arg(fVal,0,'f',3).arg(d_laserUnits);
    ui->timeTracePlot->setPlotTitle(labelText);
    ui->timeTracePlot->setData(slice);
}

void LifDisplayWidget::updateLifTrace()
{

    auto d = d_currentLifConfig.lifData();
    if(d_currentSpectrumDelayIndex < d.size())
    {
        if(d_currentTimeTraceFreqIndex < d.at(d_currentSpectrumDelayIndex).size())
        {
            auto t = d_currentLifConfig.lifData().at(d_currentSpectrumDelayIndex).at(d_currentTimeTraceFreqIndex);
            ui->lifTracePlot->traceProcessed(t);
            ui->lifTracePlot->setLifGateRange(d_currentLifConfig.lifGate().first,d_currentLifConfig.lifGate().second);
            if(t.hasRefData())
                ui->lifTracePlot->setRefGateRange(d_currentLifConfig.refGate().first,d_currentLifConfig.refGate().second);
        }
        else
            ui->lifTracePlot->clearPlot();
    }
    else
        ui->lifTracePlot->clearPlot();
}

QPair<double,double> LifDisplayWidget::integrate(const LifConfig c)
{
    d_currentIntegratedData = QVector<double>(c.numDelayPoints()*c.numLaserPoints());
    auto d = c.lifData();
    QPair<double,double> out = qMakePair(0.0,0.0);
    for(int i=0; i<d.size(); i++)
    {
        int ioffset = 0;
        if(d_currentLifConfig.delayStep() < 0.0)
            ioffset = d_currentLifConfig.numDelayPoints()-1;

        for(int j=0; j<d.at(i).size(); j++)
        {
            int joffset = 0;
            if(d_currentLifConfig.laserStep() < 0.0)
                joffset = d_currentLifConfig.numLaserPoints()-1;

            int ii = qAbs(i-ioffset)*c.numLaserPoints();
            int jj = qAbs(j-joffset);
            auto integral = d.at(i).at(j).integrate(c.lifGate().first,c.lifGate().second,c.refGate().first,c.refGate().second);
            d_currentIntegratedData[ii+jj] = integral;
            out.first = qMin(integral,out.first);
            out.second = qMax(integral,out.second);
        }
    }
    return out;
}

void LifDisplayWidget::resizeEvent(QResizeEvent *ev)
{
    int margin = 5;
    ui->lifTracePlot->setGeometry(0,0,ev->size().width()/3-margin,2*ev->size().height()/5-margin);
    ui->timeTracePlot->setGeometry(ev->size().width()/3,0,ev->size().width()/3-margin,2*ev->size().height()/5-margin);
    ui->spectrumPlot->setGeometry(2*ev->size().width()/3,0,ev->size().width()/3-margin,2*ev->size().height()/5-margin);
    ui->lifSpectrogram->setGeometry(0,2*ev->size().height()/5,ev->size().width()-margin,3*ev->size().height()/5-margin);
}
