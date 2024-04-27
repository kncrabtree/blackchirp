#include <modules/lif/gui/lifdisplaywidget.h>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>

#include <data/storage/settingsstorage.h>
#include <modules/lif/gui/lifsliceplot.h>
#include <modules/lif/gui/liftraceplot.h>
#include <modules/lif/gui/lifspectrogramplot.h>
#include <modules/lif/gui/lifprocessingwidget.h>
#include <modules/lif/hardware/liflaser/liflaser.h>
#include <math.h>

#include <qwt6/qwt_matrix_raster_data.h>

LifDisplayWidget::LifDisplayWidget(QWidget *parent) :
    QWidget(parent)
{

    SettingsStorage s(BC::Key::LifLaser::key,SettingsStorage::Hardware);

    p_laserSlicePlot = new LifSlicePlot(BC::Key::lifSpectrumPlot,this);
    p_laserSlicePlot->setPlotAxisTitle(QwtPlot::xBottom,
                                      QString("Laser Position (%1)")
                                      .arg(s.get<QString>(BC::Key::LifLaser::units,"nm")));

    p_delaySlicePlot = new LifSlicePlot(BC::Key::lifTimePlot,this);
    p_delaySlicePlot->setPlotAxisTitle(QwtPlot::xBottom,QString::fromUtf16(u"Delay (µs)"));

    p_lifTracePlot = new LifTracePlot(this);
    p_spectrogramPlot = new LifSpectrogramPlot(this);
    p_procWidget = new LifProcessingWidget(false,this);
    connect(p_procWidget,&LifProcessingWidget::settingChanged,[=](){ p_lifTracePlot->setAllProcSettings(p_procWidget->getSettings());});
    connect(p_procWidget,&LifProcessingWidget::reprocessSignal,this,&LifDisplayWidget::reprocess);
    connect(p_procWidget,&LifProcessingWidget::resetSignal,this,&LifDisplayWidget::resetProc);
    connect(p_procWidget,&LifProcessingWidget::saveSignal,this,&LifDisplayWidget::saveProc);
    p_procWidget->setEnabled(false);

    auto pgb = new QGroupBox("Processing",this);
    auto pvbl = new QVBoxLayout;
    pgb->setLayout(pvbl);
    pvbl->addWidget(p_procWidget);

    connect(p_spectrogramPlot,&LifSpectrogramPlot::laserSlice,this,&LifDisplayWidget::changeLaserSlice);
    connect(p_spectrogramPlot,&LifSpectrogramPlot::delaySlice,this,&LifDisplayWidget::changeDelaySlice);

    auto hbl = new QHBoxLayout;
    hbl->addWidget(p_lifTracePlot,1);
    hbl->addWidget(p_delaySlicePlot,1);
    hbl->addWidget(p_laserSlicePlot,1);

    auto hbl2 = new QHBoxLayout;
    hbl2->addWidget(pgb);
    hbl2->addWidget(p_spectrogramPlot,1);
    auto vbl = new QVBoxLayout;
    vbl->addLayout(hbl,2);
    vbl->addLayout(hbl2,3);

    setLayout(vbl);

}

LifDisplayWidget::~LifDisplayWidget()
{
}

void LifDisplayWidget::prepareForExperiment(const Experiment &e)
{
    p_lifTracePlot->clearPlot();
    p_delaySlicePlot->prepareForExperiment();
    p_laserSlicePlot->prepareForExperiment();
    p_spectrogramPlot->clear();
    p_procWidget->setEnabled(false);
    d_currentIntegratedData.clear();

    d_dString = QString("Delay: %1 ")+BC::Unit::us;
    d_lString = QString("Laser: %1 ");
    auto it = e.d_hardware.find(BC::Key::LifLaser::key);
    if(it != e.d_hardware.end())
    {
        SettingsStorage s(it->first,SettingsStorage::Hardware);
        d_lString.append(s.get(BC::Key::LifLaser::units,QString("nm")));
        d_lDec = s.get(BC::Key::LifLaser::decimals,2);
    }

    if(e.lifEnabled())
    {
        ps_lifStorage = e.lifConfig()->storage();
        p_spectrogramPlot->prepareForExperiment(*e.lifConfig());
        p_procWidget->initialize(e.lifConfig()->scopeConfig().d_recordLength,e.lifConfig()->scopeConfig().d_refEnabled);
        p_procWidget->setAll(e.lifConfig()->d_procSettings);
        p_lifTracePlot->setNumAverages(e.lifConfig()->d_shotsPerPoint);
        d_delayReverse = e.lifConfig()->d_delayStepUs < 0.0;
        d_laserReverse = e.lifConfig()->d_laserPosStep < 0.0;
        d_currentIntegratedData.resize(e.lifConfig()->d_delayPoints*e.lifConfig()->d_laserPosPoints);
    }
    else
    {
        ps_lifStorage.reset();
        d_delayReverse = false;
        d_laserReverse = false;
    }
}

void LifDisplayWidget::experimentComplete()
{
    p_procWidget->experimentComplete();
}

void LifDisplayWidget::updatePoint()
{
    //1. Get current trace from storage
    auto t = ps_lifStorage->currentLifTrace();

    auto lp = ps_lifStorage->d_laserPoints;
//    auto dp = ps_lifStorage->d_delayPoints;

    //2. Reverse indices if needed.
    auto di = t.delayIndex();
    if(d_delayReverse)
        di = ps_lifStorage->d_delayPoints-di-1;

    auto li = t.laserIndex();
    if(d_laserReverse)
        li = lp-li-1;

    //3. Integrate and store integral in matrix
    auto d = t.integrate(p_procWidget->getSettings());
    d_currentIntegratedData[li + di*lp] = d;

    //4. Update spectrogram and spectrogram indices
    p_spectrogramPlot->updateData(d_currentIntegratedData,lp);
    p_spectrogramPlot->setLiveIndices(di,li);

    //5. Update plots if spectrogram index matches appropriate current trace index
    auto cdi = p_spectrogramPlot->currentDelayIndex();
    auto cli = p_spectrogramPlot->currentLaserIndex();

    if(cdi == di)
        p_laserSlicePlot->setData(laserSlice(di),d_dString.arg(p_spectrogramPlot->delayVal(di),0,'f',3));
    if(cli == li)
        p_delaySlicePlot->setData(delaySlice(li),d_lString.arg(p_spectrogramPlot->laserVal(li),0,'f',d_lDec));
    if(cdi == di && cli == li)
        p_lifTracePlot->setTrace(t);

}

void LifDisplayWidget::changeLaserSlice(int di)
{
    p_laserSlicePlot->setData(laserSlice(di),d_dString.arg(p_spectrogramPlot->delayVal(di),0,'f',3));
    auto li = p_spectrogramPlot->currentLaserIndex();

    if(d_delayReverse)
        di = ps_lifStorage->d_delayPoints - di -1;
    if(d_laserReverse)
        li = ps_lifStorage->d_laserPoints - li -1;

    auto t = ps_lifStorage->getLifTrace(di,li);
    p_lifTracePlot->setTrace(t);
}

void LifDisplayWidget::changeDelaySlice(int li)
{
    p_delaySlicePlot->setData(delaySlice(li),d_lString.arg(p_spectrogramPlot->laserVal(li),0,'f',d_lDec));
    auto di = p_spectrogramPlot->currentDelayIndex();

    if(d_delayReverse)
        di = ps_lifStorage->d_delayPoints - di -1;
    if(d_laserReverse)
        li = ps_lifStorage->d_laserPoints - li -1;

    auto t = ps_lifStorage->getLifTrace(di,li);
    p_lifTracePlot->setTrace(t);
}

void LifDisplayWidget::reprocess()
{
    auto lp = ps_lifStorage->d_laserPoints;
    auto dp = ps_lifStorage->d_delayPoints;
    d_currentIntegratedData = QVector<double>(dp*lp);

    auto ps = p_procWidget->getSettings();
    for(int li=0; li<lp; li++)
    {
        auto mli = li;
        if(d_laserReverse)
            mli = lp-li-1;
        for(int di=0; di<dp; di++)
        {
            auto mdi = di;
            if(d_delayReverse)
                mdi = dp-di-1;

            auto t = ps_lifStorage->getLifTrace(di,li);
            auto d = t.integrate(ps);
            d_currentIntegratedData[mli+mdi*lp] = d;
        }
    }

    p_spectrogramPlot->updateData(d_currentIntegratedData,lp);

    auto cdi = p_spectrogramPlot->currentDelayIndex();
    auto cli = p_spectrogramPlot->currentLaserIndex();

    p_laserSlicePlot->setData(laserSlice(cdi),d_dString.arg(p_spectrogramPlot->delayVal(cdi),0,'f',3));
    p_delaySlicePlot->setData(delaySlice(cli),d_lString.arg(p_spectrogramPlot->laserVal(cli),0,'f',d_lDec));

    if(d_delayReverse)
        cdi = dp - cdi -1;
    if(d_laserReverse)
        cli = lp - cli -1;
    auto lt = ps_lifStorage->getLifTrace(cdi,cli);
    p_lifTracePlot->setTrace(lt);

}

void LifDisplayWidget::resetProc()
{
    auto l = p_procWidget->getSettings();
    if(ps_lifStorage->readProcessingSettings(l))
        p_procWidget->setAll(l);
}

void LifDisplayWidget::saveProc()
{
    ps_lifStorage->writeProcessingSettings(p_procWidget->getSettings());
}

QVector<QPointF> LifDisplayWidget::laserSlice(int delayIndex) const
{
    auto lp = ps_lifStorage->d_laserPoints;
    QVector<QPointF> out(lp);
    auto min = p_spectrogramPlot->getlMin();
    auto dx = p_spectrogramPlot->getldx();
    for(int i=0; i<out.size(); i++)
        out[i] = {min+i*dx,d_currentIntegratedData.at(i + delayIndex*lp)};

    return out;
}

QVector<QPointF> LifDisplayWidget::delaySlice(int laserIndex) const
{
    auto lp = ps_lifStorage->d_laserPoints;
    auto dp = ps_lifStorage->d_delayPoints;
    QVector<QPointF> out(dp);
    auto min = p_spectrogramPlot->getlMin();
    auto dx = p_spectrogramPlot->getldx();
    for(int i=0; i<out.size(); i++)
        out[i] = {min+i*dx,d_currentIntegratedData.at(laserIndex + i*lp)};

    return out;
}
