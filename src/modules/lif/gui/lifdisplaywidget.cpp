#include <modules/lif/gui/lifdisplaywidget.h>

#include <QHBoxLayout>
#include <QVBoxLayout>

#include <data/storage/settingsstorage.h>
#include <modules/lif/gui/lifsliceplot.h>
#include <modules/lif/gui/liftraceplot.h>
#include <modules/lif/gui/lifspectrogramplot.h>
#include <modules/lif/hardware/liflaser/liflaser.h>
#include <math.h>

#include <qwt6/qwt_matrix_raster_data.h>

LifDisplayWidget::LifDisplayWidget(QWidget *parent) :
    QWidget(parent), d_delayReverse(false), d_freqReverse(false), d_currentSpectrumDelayIndex(-1), d_currentTimeTraceFreqIndex(-1)
{

    SettingsStorage s(BC::Key::lifLaser,SettingsStorage::Hardware);

    p_freqSlicePlot = new LifSlicePlot(BC::Key::lifSpectrumPlot,this);
    p_freqSlicePlot->setPlotAxisTitle(QwtPlot::xBottom,
                                      QString("Laser Position (%1)")
                                      .arg(s.get<QString>(BC::Key::lifLaserUnits,"nm")));
    p_freqSlicePlot->setPlotTitle(QString("Laser Slice"));

    p_timeSlicePlot = new LifSlicePlot(BC::Key::lifTimePlot,this);
    p_timeSlicePlot->setPlotAxisTitle(QwtPlot::xBottom,QString::fromUtf16(u"Delay (µs)"));
    p_timeSlicePlot->setPlotTitle(QString("Time Slice"));

    p_lifTracePlot = new LifTracePlot(this);
    p_lifTracePlot->setPlotTitle(QString("Time Trace"));
    p_lifTracePlot->setDisplayOnly(true);

    p_spectrogramPlot = new LifSpectrogramPlot(this);

    connect(p_lifTracePlot,&LifTracePlot::lifGateUpdated,this,&LifDisplayWidget::lifZoneUpdate);
    connect(p_lifTracePlot,&LifTracePlot::refGateUpdated,this,&LifDisplayWidget::refZoneUpdate);
    connect(p_spectrogramPlot,&LifSpectrogramPlot::freqSlice,this,&LifDisplayWidget::freqSlice);
    connect(p_spectrogramPlot,&LifSpectrogramPlot::delaySlice,this,&LifDisplayWidget::delaySlice);

    auto hbl = new QHBoxLayout;
    hbl->addWidget(p_lifTracePlot,1);
    hbl->addWidget(p_timeSlicePlot,1);
    hbl->addWidget(p_freqSlicePlot,1);

    auto vbl = new QVBoxLayout;
    vbl->addLayout(hbl,2);
    vbl->addWidget(p_spectrogramPlot,3);

    setLayout(vbl);

}

LifDisplayWidget::~LifDisplayWidget()
{
}

void LifDisplayWidget::checkLifColors()
{
    p_lifTracePlot->checkColors();
}

void LifDisplayWidget::resetLifPlot()
{
    p_lifTracePlot->reset();
}

void LifDisplayWidget::prepareForExperiment(const Experiment &e)
{
    p_lifTracePlot->clearPlot();

    p_freqSlicePlot->setPlotTitle(QString("Laser Slice"));
    p_timeSlicePlot->setPlotTitle(QString("Time Slice"));


    p_timeSlicePlot->prepareForExperiment();
    p_freqSlicePlot->prepareForExperiment();

    if(e.lifEnabled())
    {
        d_currentTimeTraceFreqIndex = -1;
        d_currentSpectrumDelayIndex = -1;
    }
    else
    {
        d_currentTimeTraceFreqIndex = 0;
        d_currentSpectrumDelayIndex = 0;
    }

    p_spectrogramPlot->prepareForExperiment(c);
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
    p_spectrogramPlot->updateData(d_currentIntegratedData,d_currentLifConfig.numLaserPoints(),zRange.first,zRange.second);


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
    p_freqSlicePlot->setPlotTitle(labelText);
    p_freqSlicePlot->setData(slice);
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

    SettingsStorage s(BC::Key::lifLaser,SettingsStorage::Hardware);
    double fVal = d_currentLifConfig.laserRange().first + d_currentTimeTraceFreqIndex*d_currentLifConfig.laserStep();
    QString labelText = QString::fromUtf16(u"Time Slice at %1 %2").arg(fVal,0,'f',3).arg(s.get<QString>(BC::Key::lifLaserUnits));
    p_timeSlicePlot->setPlotTitle(labelText);
    p_timeSlicePlot->setData(slice);
}

void LifDisplayWidget::updateLifTrace()
{

    auto d = d_currentLifConfig.lifData();
    if(d_currentSpectrumDelayIndex < d.size())
    {
        if(d_currentTimeTraceFreqIndex < d.at(d_currentSpectrumDelayIndex).size())
        {
            auto t = d_currentLifConfig.lifData().at(d_currentSpectrumDelayIndex).at(d_currentTimeTraceFreqIndex);
            p_lifTracePlot->traceProcessed(t);
            p_lifTracePlot->setLifGateRange(d_currentLifConfig.lifGate().first,d_currentLifConfig.lifGate().second);
            if(t.hasRefData())
                p_lifTracePlot->setRefGateRange(d_currentLifConfig.refGate().first,d_currentLifConfig.refGate().second);
        }
        else
            p_lifTracePlot->clearPlot();
    }
    else
        p_lifTracePlot->clearPlot();
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
