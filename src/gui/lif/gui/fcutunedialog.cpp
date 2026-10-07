#include <gui/lif/gui/fcutunedialog.h>

#include <cmath>

#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QProgressBar>
#include <QLabel>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QShowEvent>
#include <QCloseEvent>

#include <data/lif/fcutunecontroller.h>
#include <data/lif/lifconfig.h>
#include <gui/lif/gui/lifsliceplot.h>

using namespace Qt::StringLiterals;
using namespace BC::Key::FcuTuneDialog;

FcuTuneDialog::FcuTuneDialog(const QString &digitizerHwKey, std::function<void(LifConfig&)> configProvider,
                             QWidget *parent) :
    QDialog(parent), SettingsStorage(key), d_digitizerHwKey(digitizerHwKey),
    d_configProvider(std::move(configProvider))
{
    setWindowTitle(u"FCU Phase-Match Tuning"_s);

    p_controller = new FcuTuneController(this);
    connect(p_controller,&FcuTuneController::requestTrim,this,&FcuTuneDialog::requestTrim);
    connect(p_controller,&FcuTuneController::pointComplete,this,&FcuTuneDialog::pointComplete);
    connect(p_controller,&FcuTuneController::finished,this,&FcuTuneDialog::sweepFinished);

    auto vbl = new QVBoxLayout;

    auto stageHbl = new QHBoxLayout;
    stageHbl->addWidget(new QLabel(u"Stage"_s));
    p_stageBox = new QComboBox;
    p_stageBox->setToolTip(u"Frequency-conversion stage to tune. Only stages that support a trim are listed."_s);
    stageHbl->addWidget(p_stageBox,1);
    p_trimLabel = new QLabel;
    stageHbl->addWidget(p_trimLabel);
    vbl->addLayout(stageHbl);

    auto sweepBox = new QGroupBox(u"Sweep"_s);
    auto fl = new QFormLayout;

    p_halfWidthBox = new QSpinBox;
    p_halfWidthBox->setRange(1,1000000);
    p_halfWidthBox->setSingleStep(100);
    p_halfWidthBox->setSuffix(u" steps"_s);
    p_halfWidthBox->setValue(get(halfWidth,2000));
    p_halfWidthBox->setToolTip(u"The sweep spans this far either side of the current trim."_s);
    registerGetter(halfWidth,p_halfWidthBox,&QSpinBox::value);
    fl->addRow(u"Half width"_s,p_halfWidthBox);

    p_pointsBox = new QSpinBox;
    p_pointsBox->setRange(3,201);
    p_pointsBox->setValue(get(points,11));
    registerGetter(points,p_pointsBox,&QSpinBox::value);
    fl->addRow(u"Points"_s,p_pointsBox);

    p_waveformsBox = new QSpinBox;
    p_waveformsBox->setRange(1,100000);
    p_waveformsBox->setValue(get(waveforms,20));
    p_waveformsBox->setToolTip(u"Digitizer records averaged at each point. Each record may itself "
                                "be an average of several shots (the digitizer averages setting)."_s);
    registerGetter(waveforms,p_waveformsBox,&QSpinBox::value);
    fl->addRow(u"Records/point"_s,p_waveformsBox);

    p_discardBox = new QSpinBox;
    p_discardBox->setRange(0,1000);
    p_discardBox->setValue(get(discard,2));
    p_discardBox->setToolTip(u"Records ignored after each move, before any are averaged."_s);
    registerGetter(discard,p_discardBox,&QSpinBox::value);
    fl->addRow(u"Discard after move"_s,p_discardBox);

    p_contrastBox = new QDoubleSpinBox;
    p_contrastBox->setRange(0.0,100.0);
    p_contrastBox->setDecimals(1);
    p_contrastBox->setSuffix(u" %"_s);
    p_contrastBox->setValue(get(minContrast,20.0));
    p_contrastBox->setToolTip(u"Minimum (max - min)/max of the reference signal across the sweep "
                               "for the peak fit to be accepted."_s);
    registerGetter(minContrast,p_contrastBox,&QDoubleSpinBox::value);
    fl->addRow(u"Min contrast"_s,p_contrastBox);

    sweepBox->setLayout(fl);

    auto manualBox = new QGroupBox(u"Manual Trim"_s);
    auto mhbl = new QHBoxLayout;
    p_manualTrimBox = new QSpinBox;
    p_manualTrimBox->setRange(-16777215,16777215);
    p_manualTrimBox->setSingleStep(100);
    p_manualTrimBox->setSuffix(u" steps"_s);
    mhbl->addWidget(p_manualTrimBox,1);
    p_setTrimButton = new QPushButton(u"Set"_s);
    mhbl->addWidget(p_setTrimButton);
    p_zeroTrimButton = new QPushButton(u"Zero"_s);
    p_zeroTrimButton->setToolTip(u"Return to the stored calibration."_s);
    mhbl->addWidget(p_zeroTrimButton);
    manualBox->setLayout(mhbl);

    auto controlsHbl = new QHBoxLayout;
    controlsHbl->addWidget(sweepBox,1);
    controlsHbl->addWidget(manualBox,1,Qt::AlignTop);
    vbl->addLayout(controlsHbl);

    auto runHbl = new QHBoxLayout;
    p_startButton = new QPushButton(u"Start Sweep"_s);
    p_abortButton = new QPushButton(u"Abort"_s);
    p_progressBar = new QProgressBar;
    p_progressBar->setRange(0,1000);
    p_progressBar->setValue(0);
    p_progressBar->setTextVisible(false);
    runHbl->addWidget(p_startButton);
    runHbl->addWidget(p_abortButton);
    runHbl->addWidget(p_progressBar,1);
    vbl->addLayout(runHbl);

    p_statusLabel = new QLabel;
    p_statusLabel->setWordWrap(true);
    vbl->addWidget(p_statusLabel);

    p_plot = new LifSlicePlot(QString(plot),this);
    p_plot->setPlotAxisTitle(QwtPlot::xBottom,u"Trim (steps)"_s);
    p_plot->setPlotAxisTitle(QwtPlot::yLeft,u"Reference (AU)"_s);
    p_plot->setMinimumHeight(250);
    vbl->addWidget(p_plot,1);

    setLayout(vbl);

    connect(p_controller,&FcuTuneController::progress,p_progressBar,&QProgressBar::setValue);
    connect(p_startButton,&QPushButton::clicked,this,&FcuTuneDialog::startSweep);
    connect(p_abortButton,&QPushButton::clicked,p_controller,&FcuTuneController::abort);
    connect(p_setTrimButton,&QPushButton::clicked,this,[this](){
        emit requestTrim(p_stageBox->currentText(),static_cast<double>(p_manualTrimBox->value()));
    });
    connect(p_zeroTrimButton,&QPushButton::clicked,this,[this](){
        emit requestTrim(p_stageBox->currentText(),0.0);
    });
    connect(p_stageBox,&QComboBox::currentTextChanged,this,[this](){
        updateTrimLabel();
        updateControls();
    });

    updateTrimLabel();
    updateControls();
}

FcuTuneDialog::~FcuTuneDialog() = default;

void FcuTuneDialog::setAcquiring(bool acquiring)
{
    d_acquiring = acquiring;
    if(!acquiring && p_controller->isRunning())
        p_controller->abort();
    updateControls();
}

void FcuTuneDialog::stageTrimUpdate(const QString &stageKey, double trim, int direction, bool success)
{
    auto &info = d_stages[stageKey];
    info.trim = trim;
    info.direction = direction;
    if(p_stageBox->findText(stageKey) < 0)
        p_stageBox->addItem(stageKey);

    // The controller ignores updates for other stages and while idle.
    p_controller->trimUpdated(stageKey,trim,direction,success);

    if(!success && !p_controller->isRunning())
        p_statusLabel->setText(u"Could not set the trim of %1; see the log."_s.arg(stageKey));

    updateTrimLabel();
    updateControls();
}

void FcuTuneDialog::newWaveform(const QVector<qint8> b)
{
    if(p_controller->isRunning())
        p_controller->processWaveform(b);
}

void FcuTuneDialog::showEvent(QShowEvent *e)
{
    emit requestTrimReport();
    QDialog::showEvent(e);
}

void FcuTuneDialog::closeEvent(QCloseEvent *e)
{
    p_controller->abort();
    QDialog::closeEvent(e);
}

void FcuTuneDialog::reject()
{
    p_controller->abort();
    QDialog::reject();
}

void FcuTuneDialog::startSweep()
{
    auto stage = p_stageBox->currentText();
    auto it = d_stages.find(stage);
    if(it == d_stages.end())
        return;

    LifConfig cfg(d_digitizerHwKey);
    if(d_configProvider)
        d_configProvider(cfg);

    if(!cfg.digitizerConfig().d_refEnabled)
    {
        p_statusLabel->setText(u"Enable the reference channel and set its gate before sweeping."_s);
        return;
    }

    BC::FcuTune::Settings s;
    s.halfWidth = p_halfWidthBox->value();
    s.points = p_pointsBox->value();
    s.waveformsPerPoint = p_waveformsBox->value();
    s.discardPerPoint = p_discardBox->value();
    s.minContrast = p_contrastBox->value()/100.0;
    s.direction = it->second.direction;

    d_points.clear();
    p_plot->setData(d_points);

    if(p_controller->start(stage,it->second.trim,s,cfg.digitizerConfig(),cfg.d_procSettings))
        p_statusLabel->setText(u"Sweeping %1 ± %2 steps around trim %3..."_s
                                   .arg(stage).arg(s.halfWidth,0,'f',0).arg(it->second.trim,0,'f',0));

    updateControls();
}

void FcuTuneDialog::pointComplete(double trim, double mean, double stdErr)
{
    Q_UNUSED(stdErr)
    d_points.append({trim,mean});
    p_plot->setData(d_points);
    p_plot->autoScale();
}

void FcuTuneDialog::sweepFinished(const BC::FcuTune::Result &r, double finalTrim, bool moveOk)
{
    auto txt = r.message;
    if(!moveOk)
        txt += u" The final move to trim %1 failed; see the log."_s.arg(finalTrim,0,'f',0);
    p_statusLabel->setText(txt);

    QString label;
    if(r.success())
        label = u"Peak: %1 ± %2"_s.arg(r.center,0,'f',0).arg(r.centerUncertainty,0,'f',0);
    p_plot->setData(d_points,label);

    if(!r.success())
        p_progressBar->setValue(0);

    updateControls();
}

void FcuTuneDialog::updateControls()
{
    bool running = p_controller->isRunning();
    bool haveStage = d_stages.find(p_stageBox->currentText()) != d_stages.end();

    p_stageBox->setEnabled(!running);
    p_startButton->setEnabled(!running && haveStage && d_acquiring);
    p_startButton->setToolTip(d_acquiring ? QString() : u"Start the LIF acquisition first."_s);
    p_abortButton->setEnabled(running);
    p_setTrimButton->setEnabled(!running && haveStage);
    p_zeroTrimButton->setEnabled(!running && haveStage);
    p_halfWidthBox->setEnabled(!running);
    p_pointsBox->setEnabled(!running);
    p_waveformsBox->setEnabled(!running);
    p_discardBox->setEnabled(!running);
    p_contrastBox->setEnabled(!running);
}

void FcuTuneDialog::updateTrimLabel()
{
    auto it = d_stages.find(p_stageBox->currentText());
    if(it == d_stages.end())
    {
        p_trimLabel->setText(u"No trim-capable stage"_s);
        return;
    }

    p_trimLabel->setText(u"Current trim: %1 steps"_s.arg(it->second.trim,0,'f',0));
}
