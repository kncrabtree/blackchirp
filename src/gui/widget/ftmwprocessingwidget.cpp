#include <gui/widget/ftmwprocessingwidget.h>
#include <QFormLayout>

#include <QLabel>
#include <QMenu>

FtmwProcessingWidget::FtmwProcessingWidget(QWidget *parent) :
    QWidget(parent), SettingsStorage(BC::Key::ftmwProcWidget)
{
    auto fl = new QFormLayout;

    p_startBox = new QDoubleSpinBox;
    p_startBox->setMinimum(0.0);
    p_startBox->setDecimals(4);
    p_startBox->setSingleStep(0.05);
    p_startBox->setValue(get<double>(BC::Key::fidStart,0.0));
    p_startBox->setKeyboardTracking(false);
    p_startBox->setToolTip(QString("Start of data for FT. Points before this will be set to 0."));
    auto vc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    connect(p_startBox,vc,this,&FtmwProcessingWidget::readSettings);
    p_startBox->setSuffix(QString::fromUtf8(" μs"));

    registerGetter(BC::Key::fidStart,p_startBox,&QDoubleSpinBox::value);

    auto sl = new QLabel(QString("FT Start"));
    sl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(sl,p_startBox);

    p_endBox = new QDoubleSpinBox;
    p_endBox->setMinimum(0.0);
    p_endBox->setDecimals(4);
    p_endBox->setSingleStep(0.05);
    p_endBox->setValue(get<double>(BC::Key::fidEnd,p_endBox->maximum()));
    p_endBox->setKeyboardTracking(false);
    p_endBox->setToolTip(QString("End of data for FT. Points after this will be set to 0."));
    connect(p_endBox,vc,this,&FtmwProcessingWidget::readSettings);
    p_endBox->setSuffix(QString::fromUtf8(" μs"));

    registerGetter(BC::Key::fidEnd,p_endBox,&QDoubleSpinBox::value);

    auto el = new QLabel(QString("FT End"));
    el->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(el,p_endBox);

    p_autoScaleIgnoreBox = new QDoubleSpinBox;
    p_autoScaleIgnoreBox->setRange(0.0,1000.0);
    p_autoScaleIgnoreBox->setDecimals(1);
    p_autoScaleIgnoreBox->setValue(get<double>(BC::Key::autoscaleIgnore,0.0));
    p_autoScaleIgnoreBox->setSuffix(QString(" MHz"));
    p_autoScaleIgnoreBox->setKeyboardTracking(false);
    connect(p_autoScaleIgnoreBox,vc,this,&FtmwProcessingWidget::readSettings);
    p_autoScaleIgnoreBox->setToolTip(QString("Points within this frequency of the LO are ignored when calculating the autoscale minimum and maximum."));

    registerGetter(BC::Key::autoscaleIgnore,p_autoScaleIgnoreBox,&QDoubleSpinBox::value);

    auto asl = new QLabel(QString("VScale Ignore"));
    asl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(asl,p_autoScaleIgnoreBox);

    p_zeroPadBox = new QSpinBox;
    p_zeroPadBox->setRange(0,4);
    p_zeroPadBox->setValue(get<int>(BC::Key::zeroPad));
    p_zeroPadBox->setSpecialValueText(QString("None"));
    p_zeroPadBox->setKeyboardTracking(false);
    auto ivc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    connect(p_zeroPadBox,ivc,this,&FtmwProcessingWidget::readSettings);
    p_zeroPadBox->setToolTip(QString("Pad FID with zeroes until length extends to a power of 2.\n1 = next power of 2, 2 = second power of 2, etc."));

    registerGetter(BC::Key::zeroPad,p_zeroPadBox,&QSpinBox::value);

    auto zpl = new QLabel(QString("Zero Pad"));
    zpl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(zpl,p_zeroPadBox);

    p_removeDCBox = new QCheckBox;
    p_removeDCBox->setChecked(get<bool>(BC::Key::removeDC,false));
    p_removeDCBox->setToolTip(QString("Subtract any DC offset in the FID."));
    connect(p_removeDCBox,&QCheckBox::toggled,this,&FtmwProcessingWidget::readSettings);

    registerGetter(BC::Key::removeDC,static_cast<QAbstractButton*>(p_removeDCBox),&QCheckBox::isChecked);

    auto rdcl = new QLabel(QString("Remove DC"));
    rdcl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(rdcl,p_removeDCBox);

    auto wfl = new QLabel("Window");
    wfl->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);

    p_winfBox = new QComboBox;
    p_winfBox->addItem(QString("None"),FtWorker::Boxcar);
    p_winfBox->addItem(QString("Bartlett"),FtWorker::Bartlett);
    p_winfBox->addItem(QString("Blackman"),FtWorker::Blackman);
    p_winfBox->addItem(QString("Blackman-Harris"),FtWorker::BlackmanHarris);
    p_winfBox->addItem(QString("Hamming"),FtWorker::Hamming);
    p_winfBox->addItem(QString("Hanning"),FtWorker::Hanning);
    p_winfBox->addItem(QString("Kaiser-Bessel"),FtWorker::KaiserBessel14);

    p_winfBox->setCurrentIndex(p_winfBox->findData(get<FtWorker::FtWindowFunction>(BC::Key::ftWinf,FtWorker::Boxcar)));
    connect(p_winfBox,static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),
            this,&FtmwProcessingWidget::readSettings);

    registerGetter(BC::Key::ftWinf,
                   std::function<FtWorker::FtWindowFunction ()>
    { [this] () {return this->p_winfBox->currentData().value<FtWorker::FtWindowFunction>();}});

    fl->addRow(wfl,p_winfBox);


    auto ufl = new QLabel("FT Units");
    ufl->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);


    p_unitsBox = new QComboBox;
    p_unitsBox->addItem(QString("V"),FtWorker::FtV);
    p_unitsBox->addItem(QString("mV"),FtWorker::FtmV);
    p_unitsBox->addItem(QString::fromUtf16(u"μV"),FtWorker::FtuV);
    p_unitsBox->addItem(QString("nV"),FtWorker::FtnV);
    connect(p_unitsBox,static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),
            this,&FtmwProcessingWidget::readSettings);
    fl->addRow(ufl,p_unitsBox);
    p_unitsBox->setCurrentIndex(p_unitsBox->findData(get<FtWorker::FtUnits>(BC::Key::ftUnits,FtWorker::FtV)));

    registerGetter(BC::Key::ftUnits,
                   std::function<FtWorker::FtUnits ()>
    { [this](){ return this->p_unitsBox->currentData().value<FtWorker::FtUnits>(); }});

    setLayout(fl);
}

FtmwProcessingWidget::~FtmwProcessingWidget()
{
}

FtWorker::FidProcessingSettings FtmwProcessingWidget::getSettings()
{
    double start = p_startBox->value();
    double stop = p_endBox->value();
    bool rdc = p_removeDCBox->isChecked();
    int zeroPad = p_zeroPadBox->value();
    double ignore = p_autoScaleIgnoreBox->value();
    auto units = p_unitsBox->currentData().value<FtWorker::FtUnits>();
    auto winf = p_winfBox->currentData().value<FtWorker::FtWindowFunction>();

    return { start, stop, zeroPad, rdc, units, ignore, winf };
}

void FtmwProcessingWidget::prepareForExperient(const Experiment &e)
{
    setEnabled(e.d_ftmwCfg.d_isEnabled);

    if(e.d_ftmwCfg.d_isEnabled)
    {
        p_startBox->setRange(0.0,e.d_ftmwCfg.fidDurationUs());
        p_endBox->setRange(0.0,e.d_ftmwCfg.fidDurationUs());
    }

}

void FtmwProcessingWidget::readSettings()
{
    if(signalsBlocked())
        return;

    p_startBox->setMaximum(p_endBox->value());
    p_endBox->setMinimum(p_startBox->value());

    emit settingsUpdated(getSettings());

}

