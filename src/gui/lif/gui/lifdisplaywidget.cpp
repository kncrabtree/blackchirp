#include <gui/lif/gui/lifdisplaywidget.h>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QProgressDialog>
#include <QSpinBox>
#include <QTimerEvent>
#include <QtConcurrent/QtConcurrent>

#include <data/storage/settingsstorage.h>
#include <data/storage/blackchirpcsv.h>
#include <gui/widget/clickablelabel.h>
#include <gui/lif/gui/lifsliceplot.h>
#include <gui/lif/gui/liftraceplot.h>
#include <gui/lif/gui/lifspectrogramplot.h>
#include <gui/lif/gui/lifprocessingwidget.h>
#include <gui/widget/settingstable.h>
#include <math.h>

#include <qwt6/qwt_matrix_raster_data.h>

LifDisplayWidget::LifDisplayWidget(QWidget *parent) :
    QWidget(parent), SettingsStorage(BC::Key::LifDW::lifDwKey)
{

    p_laserSlicePlot = new LifSlicePlot(BC::Key::LifDW::lifSpectrumPlot,this);
    p_laserSlicePlot->setPlotAxisTitle(QwtPlot::xBottom,"Laser Position");

    p_delaySlicePlot = new LifSlicePlot(BC::Key::LifDW::lifTimePlot,this);
    p_delaySlicePlot->setPlotAxisTitle(QwtPlot::xBottom,QString::fromUtf16(u"Delay (µs)"));

    p_lifTracePlot = new LifTracePlot(this);
    p_spectrogramPlot = new LifSpectrogramPlot(this);
    p_procWidget = new LifProcessingWidget(false,this);
    connect(p_procWidget,&LifProcessingWidget::settingChanged,this,[this](){
        p_lifTracePlot->setAllProcSettings(p_procWidget->getSettings());
    });
    connect(p_procWidget,&LifProcessingWidget::reprocessSignal,this,&LifDisplayWidget::reprocess);
    connect(p_procWidget,&LifProcessingWidget::resetSignal,this,&LifDisplayWidget::resetProc);
    connect(p_procWidget,&LifProcessingWidget::saveSignal,this,&LifDisplayWidget::saveProc);
    p_procWidget->setEnabled(false);

    // No umbrella group box: the processing widget's own SettingsTable
    // section bands (Gates / Low Pass Filter / Savitzky-Golay) and the
    // Display section below are self-titling, matching the FTMW side
    // panels. A wrapping "Processing" frame would just double-head it.
    auto lvbl = new QVBoxLayout;

    p_refreshBox = new QSpinBox(this);
    p_refreshBox->setRange(500,10000);
    p_refreshBox->setSingleStep(500);
    p_refreshBox->setValue(get(BC::Key::LifDW::refresh,500));
    registerGetter(BC::Key::LifDW::refresh,p_refreshBox,&QSpinBox::value);
    p_refreshBox->setSuffix(" ms");
    p_refreshBox->setAlignment(Qt::AlignCenter);
    p_refreshBox->setKeyboardTracking(false);
    connect(p_refreshBox,qOverload<int>(&QSpinBox::valueChanged),this,[this](int v){
        if(d_refreshTimerId >= 0)
        {
            killTimer(d_refreshTimerId);
            d_refreshTimerId = startTimer(v);
        }
    });
    p_refreshBox->setEnabled(false);

    const auto refreshTip = QString("How often the plots refresh during an acquisition.");
    p_refreshBox->setToolTip(refreshTip);
    auto db = new SettingsTable(this);
    db->setFocusPolicy(Qt::NoFocus);
    db->addSectionRow("Display");
    db->addSettingRow("Refresh Interval",p_refreshBox,refreshTip);

    lvbl->addWidget(p_procWidget,1);
    lvbl->addWidget(db,0);

    connect(p_spectrogramPlot,&LifSpectrogramPlot::laserSlice,this,&LifDisplayWidget::changeLaserSlice);
    connect(p_spectrogramPlot,&LifSpectrogramPlot::delaySlice,this,&LifDisplayWidget::changeDelaySlice);

    auto hbl = new QHBoxLayout;
    hbl->addWidget(p_lifTracePlot,1);
    hbl->addWidget(p_delaySlicePlot,1);
    hbl->addWidget(p_laserSlicePlot,1);

    auto hbl2 = new QHBoxLayout;
    hbl2->addLayout(lvbl);
    hbl2->addWidget(p_spectrogramPlot,1);
    p_exptLabel = new ClickableLabel(this);
    QFont boldFont;
    boldFont.setBold(true);
    p_exptLabel->setFont(boldFont);
    p_exptLabel->setAlignment(Qt::AlignCenter);
    p_exptLabel->setText("Experiment");

    auto vbl = new QVBoxLayout;
    vbl->addWidget(p_exptLabel,0);
    vbl->addLayout(hbl,2);
    vbl->addLayout(hbl2,3);

    setLayout(vbl);

}

LifDisplayWidget::~LifDisplayWidget()
{
}

void LifDisplayWidget::prepareForExperiment(const Experiment &e)
{
    p_exptLabel->setText(QString("Experiment %1").arg(e.d_number));
    p_exptLabel->setFolderPath(
        BlackchirpCSV::exptDir(e.d_number, e.path()).absolutePath());

    p_lifTracePlot->clearPlot();
    p_delaySlicePlot->prepareForExperiment();
    p_laserSlicePlot->prepareForExperiment();
    p_spectrogramPlot->clear();
    p_procWidget->setEnabled(false);
    d_currentIntegratedData.clear();

    // .toString() is a Qt 6.4 shim — remove in 6.5+ (QString gained a
    // QStringView operator+ overload in 6.5).
    d_dString = QString("Delay: %1 ")+BC::Unit::us.toString();
    d_lString = QString("Laser: %1 ");

    if(e.lifEnabled())
    {
        // Laser display metadata follows the on-disk experiment record
        // (units from column 6 of the LaserStart header row, decimals
        // inferred from its formatted value string) rather than the
        // viewing machine's local LifLaser settings.
        d_lString.append(e.lifConfig()->laserUnits());
        d_lDec = e.lifConfig()->laserDecimals();

        ps_lifStorage = e.lifConfig()->storage();
        p_spectrogramPlot->prepareForExperiment(*e.lifConfig());
        p_procWidget->initialize(e.lifConfig()->digitizerConfig().d_recordLength,e.lifConfig()->digitizerConfig().d_refEnabled);
        p_procWidget->setAll(e.lifConfig()->d_procSettings);
        p_lifTracePlot->setNumAverages(e.lifConfig()->d_shotsPerPoint);
        d_delayReverse = e.lifConfig()->d_delayStepUs < 0.0;
        d_laserReverse = e.lifConfig()->d_laserPosStep < 0.0;
        d_currentIntegratedData.resize(e.lifConfig()->d_delayPoints*e.lifConfig()->d_laserPosPoints);
        p_refreshBox->setEnabled(true);
        d_refreshTimerId = startTimer(p_refreshBox->value());
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
    p_refreshBox->setEnabled(false);
    if(d_refreshTimerId >= 0)
        killTimer(d_refreshTimerId);
    d_refreshTimerId = -1;
    p_procWidget->experimentComplete();
}

void LifDisplayWidget::updatePoint()
{
    //1. Get current trace from storage
    auto t = ps_lifStorage->currentLifTrace();

    auto lp = ps_lifStorage->d_laserPoints;

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
    // p_spectrogramPlot->updateData(d_currentIntegratedData,lp);
    p_spectrogramPlot->setLiveIndices(di,li);

    // //5. Update plots if spectrogram index matches appropriate current trace index
    // auto cdi = p_spectrogramPlot->currentDelayIndex();
    // auto cli = p_spectrogramPlot->currentLaserIndex();

    // if(cdi == di)
    //     p_laserSlicePlot->setData(laserSlice(di),d_dString.arg(p_spectrogramPlot->delayVal(di),0,'f',3));
    // if(cli == li)
    //     p_delaySlicePlot->setData(delaySlice(li),d_lString.arg(p_spectrogramPlot->laserVal(li),0,'f',d_lDec));
    // if(cdi == di && cli == li)
    //     p_lifTracePlot->setTrace(t);

}

void LifDisplayWidget::updatePlots()
{
    auto t = ps_lifStorage->currentLifTrace();
    auto lp = ps_lifStorage->d_laserPoints;
    
    auto di = t.delayIndex();
    if(d_delayReverse)
        di = ps_lifStorage->d_delayPoints-di-1;

    auto li = t.laserIndex();
    if(d_laserReverse)
        li = lp-li-1;
    
    //Update spectrogram and spectrogram indices
    p_spectrogramPlot->updateData(d_currentIntegratedData,lp);
    // p_spectrogramPlot->setLiveIndices(di,li);

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
    // Re-entrant guard. A reprocess pass can take seconds on a
    // sweep with thousands of traces; ignore additional triggers
    // while one is in flight. The active watcher resets itself
    // when the finished slot fires, so subsequent reprocess
    // requests proceed normally.
    if(p_reprocessWatcher)
        return;

    if(!ps_lifStorage)
        return;

    const int lp = ps_lifStorage->d_laserPoints;
    const int dp = ps_lifStorage->d_delayPoints;
    const int total = dp * lp;
    if(total <= 0)
        return;

    const bool delayRev = d_delayReverse;
    const bool laserRev = d_laserReverse;
    auto storage = ps_lifStorage; // shared_ptr copy keeps storage alive
    const auto procSettings = p_procWidget->getSettings();

    // Worker: integrate every (di, li) cell off the main thread.
    // QPromise carries cancel state both directions: the watcher's
    // cancel() flips isCanceled() inside the lambda so the loop can
    // bail; the lambda's setProgressValue drives the dialog.
    auto worker = [storage, procSettings, dp, lp, delayRev, laserRev]
                  (QPromise<QVector<double>> &promise)
    {
        promise.setProgressRange(0, dp * lp);
        QVector<double> data(dp * lp);
        int progress = 0;
        for(int li = 0; li < lp; ++li)
        {
            const int mli = laserRev ? lp - li - 1 : li;
            for(int di = 0; di < dp; ++di)
            {
                if(promise.isCanceled())
                    return;
                const int mdi = delayRev ? dp - di - 1 : di;
                auto t = storage->getLifTrace(di, li);
                data[mli + mdi * lp] = t.integrate(procSettings);
                promise.setProgressValue(++progress);
            }
        }
        promise.addResult(data);
    };

    p_reprocessWatcher = new QFutureWatcher<QVector<double>>(this);

    auto *dlg = new QProgressDialog(QString("Loading LIF traces…"),
                                    QString("Cancel"),
                                    0, total, this);
    dlg->setWindowTitle(QString("Processing LIF data"));
    dlg->setWindowModality(Qt::WindowModal);
    // Suppress the dialog for fast loads (small grids land in tens of
    // milliseconds even on cold cache).
    dlg->setMinimumDuration(200);

    connect(p_reprocessWatcher, &QFutureWatcher<QVector<double>>::progressRangeChanged,
            dlg, &QProgressDialog::setRange);
    connect(p_reprocessWatcher, &QFutureWatcher<QVector<double>>::progressValueChanged,
            dlg, &QProgressDialog::setValue);
    connect(dlg, &QProgressDialog::canceled,
            p_reprocessWatcher, &QFutureWatcher<QVector<double>>::cancel);

    connect(p_reprocessWatcher, &QFutureWatcher<QVector<double>>::finished,
            this, [this, dlg, lp, dp]()
    {
        if(!p_reprocessWatcher->isCanceled())
        {
            d_currentIntegratedData = p_reprocessWatcher->result();
            p_spectrogramPlot->updateData(d_currentIntegratedData, lp);

            auto cdi = p_spectrogramPlot->currentDelayIndex();
            auto cli = p_spectrogramPlot->currentLaserIndex();

            p_laserSlicePlot->setData(laserSlice(cdi),
                d_dString.arg(p_spectrogramPlot->delayVal(cdi), 0, 'f', 3));
            p_delaySlicePlot->setData(delaySlice(cli),
                d_lString.arg(p_spectrogramPlot->laserVal(cli), 0, 'f', d_lDec));

            if(d_delayReverse)
                cdi = dp - cdi - 1;
            if(d_laserReverse)
                cli = lp - cli - 1;
            auto lt = ps_lifStorage->getLifTrace(cdi, cli);
            p_lifTracePlot->setTrace(lt);
        }

        dlg->close();
        dlg->deleteLater();
        p_reprocessWatcher->deleteLater();
        p_reprocessWatcher = nullptr;
    });

    p_reprocessWatcher->setFuture(QtConcurrent::run(worker));
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
    auto min = p_spectrogramPlot->getdMin();
    auto dx = p_spectrogramPlot->getddx();
    for(int i=0; i<out.size(); i++)
        out[i] = {min+i*dx,d_currentIntegratedData.at(laserIndex + i*lp)};

    return out;
}

void LifDisplayWidget::timerEvent(QTimerEvent *event)
{
    if(event->timerId() == d_refreshTimerId)
    {
        event->accept();
        updatePlots();
    }
}
