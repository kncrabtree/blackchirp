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

FtmwProcessingToolBar::FtmwProcessingToolBar(QWidget *parent) :
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

    registerGetter(BC::Key::fidEnd,p_endBox,&DoubleSpinBoxWidgetAction::value);
    addAction(p_endBox);


    p_autoScaleIgnoreBox = new DoubleSpinBoxWidgetAction("VScale Ignore",this);
    p_autoScaleIgnoreBox->setRange(0.0,1000.0);
    p_autoScaleIgnoreBox->setDecimals(1);
    p_autoScaleIgnoreBox->setValue(get<double>(BC::Key::autoscaleIgnore,0.0));
    p_autoScaleIgnoreBox->setSuffix(QString(" MHz"));
    connect(p_autoScaleIgnoreBox,&DoubleSpinBoxWidgetAction::valueChanged,this,&FtmwProcessingToolBar::readSettings);
    p_autoScaleIgnoreBox->setToolTip(QString("Points within this frequency of the LO are ignored when calculating the autoscale minimum and maximum."));

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
    p_removeDCBox->setChecked(get(BC::Key::removeDC,false));
    p_removeDCBox->setToolTip(QString("Subtract any DC offset in the FID."));
    connect(p_removeDCBox,&CheckWidgetAction::toggled,this,&FtmwProcessingToolBar::readSettings);

    registerGetter(BC::Key::removeDC,static_cast<QAction*>(p_removeDCBox),&CheckWidgetAction::isChecked);
    addAction(p_removeDCBox);


    p_winfBox = new EnumComboBoxWidgetAction<FtWorker::FtWindowFunction>("Window Function",this);
    p_winfBox->setValue(get(BC::Key::ftWinf,FtWorker::None));
    connect(p_winfBox,&EnumComboBoxWidgetAction<FtWorker::FtWindowFunction>::valueChanged,
            this,&FtmwProcessingToolBar::readSettings);

    registerGetter(BC::Key::ftWinf,p_winfBox,&EnumComboBoxWidgetAction<FtWorker::FtWindowFunction>::value);
    addAction(p_winfBox);


    p_unitsBox = new EnumComboBoxWidgetAction<FtWorker::FtUnits>("FT Units",this);
    p_unitsBox->setValue(get(BC::Key::ftUnits,FtWorker::FtuV));
    connect(p_unitsBox,&EnumComboBoxWidgetAction<FtWorker::FtUnits>::valueChanged,
            this,&FtmwProcessingToolBar::readSettings);

    registerGetter(BC::Key::ftUnits,p_unitsBox,&EnumComboBoxWidgetAction<FtWorker::FtUnits>::value);
    addAction(p_unitsBox);

}

FtmwProcessingToolBar::~FtmwProcessingToolBar()
{
}

FtWorker::FidProcessingSettings FtmwProcessingToolBar::getSettings()
{
    double start = p_startBox->value();
    double stop = p_endBox->value();
    bool rdc = p_removeDCBox->isChecked();
    int zeroPad = p_zeroPadBox->value();
    double ignore = p_autoScaleIgnoreBox->value();
    auto units = p_unitsBox->value();
    auto winf = p_winfBox->value();

    save();

    return { start, stop, zeroPad, rdc, units, ignore, winf };
}

void FtmwProcessingToolBar::prepareForExperient(const Experiment &e)
{
    if(e.ftmwEnabled())
    {
        p_startBox->setRange(0.0,e.ftmwConfig()->fidDurationUs());
        p_endBox->setRange(0.0,e.ftmwConfig()->fidDurationUs());
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

