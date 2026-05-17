#include <gui/widget/ftmwprocessingpanel.h>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QPushButton>
#include <QLineEdit>
#include <QMetaEnum>

#include <gui/style/themecolors.h>
#include <gui/widget/settingstable.h>

using namespace Qt::StringLiterals;

namespace {

QComboBox *makeCenteredCombo()
{
    auto *cb = new QComboBox;
    cb->setEditable(true);
    cb->lineEdit()->setReadOnly(true);
    cb->lineEdit()->setAlignment(Qt::AlignCenter);
    return cb;
}

void recenterCombo(QComboBox *cb)
{
    for(int i=0; i<cb->count(); ++i)
        cb->setItemData(i,Qt::AlignCenter,Qt::TextAlignmentRole);
}

}

FtmwProcessingPanel::FtmwProcessingPanel(bool mainWin, QWidget *parent) :
    QWidget(parent), SettingsStorage(BC::Key::ftmwProcWidget)
{
    p_table = new SettingsTable(this);
    p_table->setFocusPolicy(Qt::NoFocus);

    p_startBox = new QDoubleSpinBox;
    p_startBox->setMinimum(0.0);
    p_startBox->setDecimals(4);
    p_startBox->setSingleStep(0.05);
    p_startBox->setSuffix(u" μs"_s);
    p_startBox->setValue(get<double>(BC::Key::fidStart,0.0));
    p_startBox->setAlignment(Qt::AlignCenter);
    p_startBox->setKeyboardTracking(false);
    {
        const auto tip = "Start of data for FT. Points before this in the FID will be set to 0."_L1;
        p_startBox->setToolTip(tip);
        p_table->addSettingRow("FT Start"_L1,p_startBox,tip);
    }
    if(mainWin)
        registerGetter(BC::Key::fidStart,p_startBox,&QDoubleSpinBox::value);

    p_endBox = new QDoubleSpinBox;
    p_endBox->setMinimum(0.0);
    p_endBox->setDecimals(4);
    p_endBox->setSingleStep(0.05);
    p_endBox->setSuffix(u" μs"_s);
    p_endBox->setValue(get<double>(BC::Key::fidEnd,99.0));
    p_endBox->setAlignment(Qt::AlignCenter);
    p_endBox->setKeyboardTracking(false);
    {
        const auto tip = "End of data for FT. Points after this in the FID will be set to 0."_L1;
        p_endBox->setToolTip(tip);
        p_table->addSettingRow("FT End"_L1,p_endBox,tip);
    }
    if(mainWin)
        registerGetter(BC::Key::fidEnd,p_endBox,&QDoubleSpinBox::value);

    p_expBox = new QDoubleSpinBox;
    p_expBox->setMinimum(0.0);
    p_expBox->setDecimals(1);
    p_expBox->setSingleStep(0.1);
    p_expBox->setSpecialValueText("Disabled"_L1);
    p_expBox->setSuffix(u" μs"_s);
    p_expBox->setValue(get<double>(BC::Key::fidExp,0.0));
    p_expBox->setAlignment(Qt::AlignCenter);
    p_expBox->setKeyboardTracking(false);
    {
        const auto tip = "Time constant for an exponential filter applied to the FID. The special value 0 disables the filter."_L1;
        p_expBox->setToolTip(tip);
        p_table->addSettingRow("Exp Filter"_L1,p_expBox,tip);
    }
    if(mainWin)
        registerGetter(BC::Key::fidExp,p_expBox,&QDoubleSpinBox::value);

    p_autoScaleIgnoreBox = new QDoubleSpinBox;
    p_autoScaleIgnoreBox->setRange(0.0,1000.0);
    p_autoScaleIgnoreBox->setDecimals(1);
    p_autoScaleIgnoreBox->setSuffix(" MHz"_L1);
    p_autoScaleIgnoreBox->setValue(get<double>(BC::Key::autoscaleIgnore,0.0));
    p_autoScaleIgnoreBox->setAlignment(Qt::AlignCenter);
    p_autoScaleIgnoreBox->setKeyboardTracking(false);
    {
        const auto tip = "Points within this frequency of the LO are ignored when computing the autoscale vertical minimum and maximum."_L1;
        p_autoScaleIgnoreBox->setToolTip(tip);
        p_table->addSettingRow("VScale Ignore"_L1,p_autoScaleIgnoreBox,tip);
    }
    if(mainWin)
        registerGetter(BC::Key::autoscaleIgnore,p_autoScaleIgnoreBox,&QDoubleSpinBox::value);

    p_zeroPadBox = new QSpinBox;
    p_zeroPadBox->setRange(0,2);
    p_zeroPadBox->setSpecialValueText("None"_L1);
    p_zeroPadBox->setValue(get(BC::Key::zeroPad,0));
    p_zeroPadBox->setAlignment(Qt::AlignCenter);
    p_zeroPadBox->setKeyboardTracking(false);
    {
        const auto tip = "Pad the FID with zeroes until its length reaches a power of 2.\n1 = next power of 2, 2 = second power of 2, etc. Interpolates the spectrum; does not add information."_L1;
        p_zeroPadBox->setToolTip(tip);
        p_table->addSettingRow("Zero Pad"_L1,p_zeroPadBox,tip);
    }
    registerGetter(BC::Key::zeroPad,p_zeroPadBox,&QSpinBox::value);

    p_removeDCBox = new QCheckBox;
    p_removeDCBox->setChecked(get(BC::Key::removeDC,false));
    {
        const auto tip = "Subtract the mean (DC offset) from the FID before processing, suppressing the spurious feature at 0 Hz."_L1;
        p_removeDCBox->setToolTip(tip);
        p_table->addSettingRow("Remove DC"_L1,p_removeDCBox,tip);
    }
    if(mainWin)
        registerGetter<bool>(BC::Key::removeDC,std::function<bool()>{
            [this](){ return p_removeDCBox->isChecked(); }});

    p_winfBox = makeCenteredCombo();
    {
        auto me = QMetaEnum::fromType<FtWorker::FtWindowFunction>();
        for(int i=0; i<me.keyCount(); ++i)
            p_winfBox->addItem(QString::fromLatin1(me.key(i)),
                               QVariant::fromValue<FtWorker::FtWindowFunction>(static_cast<FtWorker::FtWindowFunction>(me.value(i))));
    }
    recenterCombo(p_winfBox);
    p_winfBox->setCurrentIndex(p_winfBox->findData(QVariant::fromValue(get(BC::Key::ftWinf,FtWorker::None))));
    {
        const auto tip = "Window function applied to the FID before the FFT. Reduces spectral leakage at the cost of frequency resolution; None is a rectangular (uniform) window."_L1;
        p_winfBox->setToolTip(tip);
        p_table->addSettingRow("Window"_L1,p_winfBox,tip);
    }
    if(mainWin)
        registerGetter<FtWorker::FtWindowFunction>(BC::Key::ftWinf,std::function<FtWorker::FtWindowFunction()>{
            [this](){ return p_winfBox->currentData().value<FtWorker::FtWindowFunction>(); }});

    p_unitsBox = makeCenteredCombo();
    {
        auto me = QMetaEnum::fromType<FtWorker::FtUnits>();
        for(int i=0; i<me.keyCount(); ++i)
            p_unitsBox->addItem(QString::fromLatin1(me.key(i)),
                                QVariant::fromValue<FtWorker::FtUnits>(static_cast<FtWorker::FtUnits>(me.value(i))));
    }
    recenterCombo(p_unitsBox);
    p_unitsBox->setCurrentIndex(p_unitsBox->findData(QVariant::fromValue(get(BC::Key::ftUnits,FtWorker::FtuV))));
    {
        const auto tip = "Amplitude units for the magnitude spectrum (volts, millivolts, microvolts, or nanovolts)."_L1;
        p_unitsBox->setToolTip(tip);
        p_table->addSettingRow("FT Units"_L1,p_unitsBox,tip);
    }
    if(mainWin)
        registerGetter<FtWorker::FtUnits>(BC::Key::ftUnits,std::function<FtWorker::FtUnits()>{
            [this](){ return p_unitsBox->currentData().value<FtWorker::FtUnits>(); }});

    p_resetButton = new QPushButton(ThemeColors::createThemedIcon(":/icons/arrow-path.svg",ThemeColors::IconSecondary,this),
                                    "Reset"_L1);
    p_resetButton->setToolTip("Reset processing settings to last saved values."_L1);
    p_saveButton = new QPushButton(ThemeColors::createThemedIcon(":/icons/arrow-down-tray.svg",ThemeColors::IconSecondary,this),
                                   "Save"_L1);
    p_saveButton->setToolTip("Save current processing settings."_L1);

    auto *btnRow = new QHBoxLayout;
    btnRow->setContentsMargins(0,0,0,0);
    btnRow->addWidget(p_resetButton);
    btnRow->addWidget(p_saveButton);

    auto *outer = new QVBoxLayout;
    outer->setContentsMargins(4,4,4,4);
    outer->addWidget(p_table,0);
    outer->addLayout(btnRow,0);
    setLayout(outer);

    auto onDouble = [this](){ readSettings(); };
    auto onInt = [this](){ readSettings(); };
    connect(p_startBox, qOverload<double>(&QDoubleSpinBox::valueChanged), this, onDouble);
    connect(p_endBox, qOverload<double>(&QDoubleSpinBox::valueChanged), this, onDouble);
    connect(p_expBox, qOverload<double>(&QDoubleSpinBox::valueChanged), this, onDouble);
    connect(p_autoScaleIgnoreBox, qOverload<double>(&QDoubleSpinBox::valueChanged), this, onDouble);
    connect(p_zeroPadBox, qOverload<int>(&QSpinBox::valueChanged), this, onInt);
    connect(p_removeDCBox, &QCheckBox::toggled, this, [this](bool){ readSettings(); });
    connect(p_winfBox, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){ readSettings(); });
    connect(p_unitsBox, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int){ readSettings(); });
    connect(p_resetButton, &QPushButton::clicked, this, [this](){ emit resetSignal(); });
    connect(p_saveButton, &QPushButton::clicked, this, [this](){ emit saveSignal(); });

    if(!mainWin)
        discardChanges();
}

FtmwProcessingPanel::~FtmwProcessingPanel() = default;

FtWorker::FidProcessingSettings FtmwProcessingPanel::getSettings()
{
    double start = p_startBox->value();
    double stop = p_endBox->value();
    double expf = p_expBox->value();
    bool rdc = p_removeDCBox->isChecked();
    int zeroPad = p_zeroPadBox->value();
    double ignore = p_autoScaleIgnoreBox->value();
    auto units = p_unitsBox->currentData().value<FtWorker::FtUnits>();
    auto winf = p_winfBox->currentData().value<FtWorker::FtWindowFunction>();

    save();

    return { start, stop, expf, zeroPad, rdc, units, ignore, winf };
}

void FtmwProcessingPanel::setAll(const FtWorker::FidProcessingSettings &c)
{
    auto b = signalsBlocked();
    blockSignals(true);
    p_startBox->setValue(c.startUs);
    p_endBox->setValue(c.endUs);
    p_expBox->setValue(c.expFilter);
    p_removeDCBox->setChecked(c.removeDC);
    p_zeroPadBox->setValue(c.zeroPadFactor);
    p_autoScaleIgnoreBox->setValue(c.autoScaleIgnoreMHz);
    p_unitsBox->setCurrentIndex(p_unitsBox->findData(QVariant::fromValue(c.units)));
    p_winfBox->setCurrentIndex(p_winfBox->findData(QVariant::fromValue(c.windowFunction)));
    blockSignals(b);

    emit settingsUpdated(getSettings());
}

void FtmwProcessingPanel::prepareForExperient(const Experiment &e)
{
    if(e.ftmwEnabled())
    {
        p_startBox->setRange(0.0,e.ftmwConfig()->fidDurationUs());
        p_endBox->setRange(0.0,e.ftmwConfig()->fidDurationUs());
        p_expBox->setRange(0.0,10.0*(e.ftmwConfig()->fidDurationUs()));

        p_resetButton->setEnabled(e.d_number > 0);
        p_saveButton->setEnabled(e.d_number > 0);

        FtWorker::FidProcessingSettings s;
        if(e.ftmwConfig()->storage()->readProcessingSettings(s))
            setAll(s);
        else
        {
            if(e.d_number > 0)
                e.ftmwConfig()->storage()->writeProcessingSettings(getSettings());
        }
    }

    setEnabled(e.ftmwEnabled());
}

void FtmwProcessingPanel::readSettings()
{
    if(signalsBlocked())
        return;

    p_startBox->setMaximum(p_endBox->value());
    p_endBox->setMinimum(p_startBox->value());

    emit settingsUpdated(getSettings());
}
