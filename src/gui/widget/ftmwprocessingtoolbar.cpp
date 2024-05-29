#include <gui/widget/ftmwprocessingtoolbar.h>
#include <QFormLayout>

#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QToolButton>
#include <QWidgetAction>
#include <QLabel>
#include <QMenu>

#include <gui/widget/toolbarwidgetaction.h>

FtmwProcessingToolBar::FtmwProcessingToolBar(bool mainWin, QWidget *parent) :
    QToolBar(parent), SettingsStorage(BC::Key::ftmwProcWidget)
{
//    layout()->setSpacing(5);
//    layout()->setAlignment(Qt::AlignRight);

    p_startBox = new DoubleSpinBoxWidgetAction("FT Start",this);
    p_startBox->setMinimum(0.0);
    p_startBox->setDecimals(4);
    p_startBox->setSingleStep(0.05);
    p_startBox->setValue(get<double>(BC::Key::fidStart,0.0));
    p_startBox->setToolTip(QString("Start of data for FT. Points before this in the FID will be set to 0."));
    p_startBox->setSuffix(QString::fromUtf8(" μs"));
    connect(p_startBox,&DoubleSpinBoxWidgetAction::valueChanged,this,&FtmwProcessingToolBar::readSettings);

    if(mainWin)
        registerGetter(BC::Key::fidStart,p_startBox,&DoubleSpinBoxWidgetAction::value);
    addAction(p_startBox);

    p_endBox = new DoubleSpinBoxWidgetAction("FT End",this);
    p_endBox->setMinimum(0.0);
    p_endBox->setDecimals(4);
    p_endBox->setSingleStep(0.05);
    p_endBox->setValue(get<double>(BC::Key::fidEnd,99.0));
    p_endBox->setToolTip(QString("End of data for FT. Points after this in the FID will be set to 0."));
    connect(p_endBox,&DoubleSpinBoxWidgetAction::valueChanged,this,&FtmwProcessingToolBar::readSettings);
    p_endBox->setSuffix(QString::fromUtf8(" μs"));

    if(mainWin)
        registerGetter(BC::Key::fidEnd,p_endBox,&DoubleSpinBoxWidgetAction::value);

    addAction(p_endBox);

    p_expBox = new DoubleSpinBoxWidgetAction("Exp Filter",this);
    p_expBox->setMinimum(0.0);
    p_expBox->setDecimals(1);
    p_expBox->setSingleStep(0.1);
    p_expBox->setSpecialValueText("Disabled");
    p_expBox->setValue(get<double>(BC::Key::fidExp,0.0));
    p_expBox->setToolTip("Time constant for exponential filter.");
    connect(p_expBox,&DoubleSpinBoxWidgetAction::valueChanged,this,&FtmwProcessingToolBar::readSettings);
    p_expBox->setSuffix(QString::fromUtf8(" μs"));

    if(mainWin)
        registerGetter(BC::Key::fidExp,p_expBox,&DoubleSpinBoxWidgetAction::value);
    addAction(p_expBox);


    p_autoScaleIgnoreBox = new DoubleSpinBoxWidgetAction("VScale Ignore",this);
    p_autoScaleIgnoreBox->setRange(0.0,1000.0);
    p_autoScaleIgnoreBox->setDecimals(1);
    p_autoScaleIgnoreBox->setValue(get<double>(BC::Key::autoscaleIgnore,0.0));
    p_autoScaleIgnoreBox->setSuffix(QString(" MHz"));
    connect(p_autoScaleIgnoreBox,&DoubleSpinBoxWidgetAction::valueChanged,this,&FtmwProcessingToolBar::readSettings);
    p_autoScaleIgnoreBox->setToolTip(QString("Points within this frequency of the LO are ignored when calculating the autoscale minimum and maximum."));

    if(mainWin)
        registerGetter(BC::Key::autoscaleIgnore,p_autoScaleIgnoreBox,&DoubleSpinBoxWidgetAction::value);

    addAction(p_autoScaleIgnoreBox);


    p_zeroPadBox = new SpinBoxWidgetAction("Zero Pad",this);
    p_zeroPadBox->setRange(0,2);
    p_zeroPadBox->setValue(get(BC::Key::zeroPad,0));
    p_zeroPadBox->setSpecialValueText("None");
    p_zeroPadBox->setToolTip("Pad FID with zeroes until length extends to a power of 2.\n1 = next power of 2, 2 = second power of 2, etc.");
    connect(p_zeroPadBox,&SpinBoxWidgetAction::valueChanged,this,&FtmwProcessingToolBar::readSettings);
    registerGetter(BC::Key::zeroPad,p_zeroPadBox,&SpinBoxWidgetAction::value);
    addAction(p_zeroPadBox);

    p_removeDCBox = new CheckWidgetAction("Remove DC",this);
    p_removeDCBox->setText("Remove DC");
    p_removeDCBox->setCheckedState(get(BC::Key::removeDC,false));
    p_removeDCBox->setToolTip(QString("Subtract any DC offset in the FID."));
    connect(p_removeDCBox,&CheckWidgetAction::checkStateChanged,this,&FtmwProcessingToolBar::readSettings);

    if(mainWin)
        registerGetter(BC::Key::removeDC,static_cast<QAction*>(p_removeDCBox),&CheckWidgetAction::isChecked);
    addAction(p_removeDCBox);


    p_winfBox = new EnumComboBoxWidgetAction<FtWorker::FtWindowFunction>("Window Function",this);
    p_winfBox->setValue(get(BC::Key::ftWinf,FtWorker::None));
    connect(p_winfBox,&EnumComboBoxWidgetAction<FtWorker::FtWindowFunction>::valueChanged,
            this,&FtmwProcessingToolBar::readSettings);

    if(mainWin)
        registerGetter(BC::Key::ftWinf,p_winfBox,&EnumComboBoxWidgetAction<FtWorker::FtWindowFunction>::value);
    addAction(p_winfBox);


    p_unitsBox = new EnumComboBoxWidgetAction<FtWorker::FtUnits>("FT Units",this);
    p_unitsBox->setValue(get(BC::Key::ftUnits,FtWorker::FtuV));
    connect(p_unitsBox,&EnumComboBoxWidgetAction<FtWorker::FtUnits>::valueChanged,
            this,&FtmwProcessingToolBar::readSettings);

    if(mainWin)
        registerGetter(BC::Key::ftUnits,p_unitsBox,&EnumComboBoxWidgetAction<FtWorker::FtUnits>::value);
    addAction(p_unitsBox);

    p_resetButton = addAction(QIcon(":/icons/reset.svg"),"Reset",this,[this](){emit resetSignal();});
    p_resetButton->setToolTip("Reset processing settings to last saved values.");
    p_saveButton = addAction(QIcon(":/icons/save-as.svg"),"Save",this,[this](){emit saveSignal();});
    p_saveButton->setToolTip("Save current processing settings.");

    if(!mainWin)
        discardChanges();
}

FtmwProcessingToolBar::~FtmwProcessingToolBar()
{
}

FtWorker::FidProcessingSettings FtmwProcessingToolBar::getSettings()
{
    double start = p_startBox->value();
    double stop = p_endBox->value();
    double expf = p_expBox->value();
    bool rdc = p_removeDCBox->readCheckedState();
    int zeroPad = p_zeroPadBox->value();
    double ignore = p_autoScaleIgnoreBox->value();
    auto units = p_unitsBox->value();
    auto winf = p_winfBox->value();

    save();

    return { start, stop, expf, zeroPad, rdc, units, ignore, winf };
}

void FtmwProcessingToolBar::setAll(const FtWorker::FidProcessingSettings &c)
{
    auto b = signalsBlocked();
    blockSignals(true);
    p_startBox->setValue(c.startUs);
    p_endBox->setValue(c.endUs);
    p_expBox->setValue(c.expFilter);
    p_removeDCBox->setCheckedState(c.removeDC);
    p_zeroPadBox->setValue(c.zeroPadFactor);
    p_autoScaleIgnoreBox->setValue(c.autoScaleIgnoreMHz);
    p_unitsBox->setCurrentValue(c.units);
    p_winfBox->setCurrentValue(c.windowFunction);
    blockSignals(b);

    emit settingsUpdated(getSettings());
}

void FtmwProcessingToolBar::prepareForExperient(const Experiment &e)
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

void FtmwProcessingToolBar::readSettings()
{
    if(signalsBlocked())
        return;

    p_startBox->setMaximum(p_endBox->value());
    p_endBox->setMinimum(p_startBox->value());

    emit settingsUpdated(getSettings());

}

