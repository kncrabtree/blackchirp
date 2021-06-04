#include <src/gui/widget/ftmwprocessingwidget.h>
#include <QFormLayout>

#include <QLabel>
#include <QMenu>
#include <QFrame>
#include <QButtonGroup>

FtmwProcessingWidget::FtmwProcessingWidget(QWidget *parent) : QWidget(parent)
{
    auto fl = new QFormLayout;

    p_startBox = new QDoubleSpinBox;
    p_startBox->setMinimum(0.0);
    p_startBox->setDecimals(4);
    p_startBox->setSingleStep(0.05);
    p_startBox->setValue(0.0);
    p_startBox->setKeyboardTracking(false);
    p_startBox->setToolTip(QString("tart of data for FT. Points before this will be set to 0."));
    auto vc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    connect(p_startBox,vc,this,&FtmwProcessingWidget::readSettings);
    p_startBox->setSuffix(QString::fromUtf8(" μs"));

    auto sl = new QLabel(QString("FT Start"));
    sl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(sl,p_startBox);

    p_endBox = new QDoubleSpinBox;
    p_endBox->setMinimum(0.0);
    p_endBox->setDecimals(4);
    p_endBox->setSingleStep(0.05);
    p_endBox->setValue(p_endBox->maximum());
    p_endBox->setKeyboardTracking(false);
    p_endBox->setToolTip(QString("End of data for FT. Points after this will be set to 0."));
    connect(p_endBox,vc,this,&FtmwProcessingWidget::readSettings);
    p_endBox->setSuffix(QString::fromUtf8(" μs"));

    auto el = new QLabel(QString("FT End"));
    el->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(el,p_endBox);

    p_autoScaleIgnoreBox = new QDoubleSpinBox;
    p_autoScaleIgnoreBox->setRange(0.0,1000.0);
    p_autoScaleIgnoreBox->setDecimals(1);
    p_autoScaleIgnoreBox->setValue(0.0);
    p_autoScaleIgnoreBox->setSuffix(QString(" MHz"));
    p_autoScaleIgnoreBox->setKeyboardTracking(false);
    connect(p_autoScaleIgnoreBox,vc,this,&FtmwProcessingWidget::readSettings);
    p_autoScaleIgnoreBox->setToolTip(QString("Points within this frequency of the LO are ignored when calculating the autoscale minimum and maximum."));

    auto asl = new QLabel(QString("VScale Ignore"));
    asl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(asl,p_autoScaleIgnoreBox);

    p_zeroPadBox = new QSpinBox;
    p_zeroPadBox->setRange(0,4);
    p_zeroPadBox->setValue(0);
    p_zeroPadBox->setSpecialValueText(QString("None"));
    p_zeroPadBox->setKeyboardTracking(false);
    auto ivc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    connect(p_zeroPadBox,ivc,this,&FtmwProcessingWidget::readSettings);
    p_zeroPadBox->setToolTip(QString("Pad FID with zeroes until length extends to a power of 2.\n1 = next power of 2, 2 = second power of 2, etc."));

    auto zpl = new QLabel(QString("Zero Pad"));
    zpl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(zpl,p_zeroPadBox);

    p_removeDCBox = new QCheckBox;
    p_removeDCBox->setToolTip(QString("Subtract any DC offset in the FID."));
    connect(p_removeDCBox,&QCheckBox::toggled,this,&FtmwProcessingWidget::readSettings);


    auto rdcl = new QLabel(QString("Remove DC"));
    rdcl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(rdcl,p_removeDCBox);

    auto f = new QFrame;
    f->setFixedHeight(3);
    f->setFrameStyle(QFrame::HLine);
    f->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Fixed);
//    f->setFrameShadow(QFrame::Sunken);
    f->setLineWidth(1);
//    f->setContentsMargins(0,1,0,1);
    fl->addRow(f);

    auto wfl = new QLabel("FT Window Function");
    wfl->setAlignment(Qt::AlignCenter);
    wfl->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);
    fl->addRow(wfl);

    d_windowTypes.insert(BlackChirp::Bartlett,QString("Bartlett"));
    d_windowTypes.insert(BlackChirp::Boxcar,QString("Boxcar (None)"));
    d_windowTypes.insert(BlackChirp::Blackman,QString("Blackman"));
    d_windowTypes.insert(BlackChirp::BlackmanHarris,QString("Blackman-Harris"));
    d_windowTypes.insert(BlackChirp::Hamming,QString("Hamming"));
    d_windowTypes.insert(BlackChirp::Hanning,QString("Hanning"));
    d_windowTypes.insert(BlackChirp::KaiserBessel14,QString("Kaiser-Bessel"));

    auto winfGroup = new QButtonGroup;
    winfGroup->setExclusive(true);
    auto bc = static_cast<void(QButtonGroup::*)(QAbstractButton *)>(&QButtonGroup::buttonClicked);
    connect(winfGroup,bc,this,&FtmwProcessingWidget::readSettings);

    for(auto it=d_windowTypes.constBegin(); it!=d_windowTypes.constEnd(); it++)
    {
        auto button = new QRadioButton;
        if(it.key() == BlackChirp::Boxcar)
            button->setChecked(true);
        winfGroup->addButton(button);

        d_windowButtons.insert(it.key(),button);
        auto label = new QLabel(it.value());
        label->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
        fl->addRow(label,button);

    }

    auto f2 = new QFrame;
    f2->setFixedHeight(3);
    f2->setFrameStyle(QFrame::HLine);
    f2->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Fixed);
//    f->setFrameShadow(QFrame::Sunken);
    f2->setLineWidth(1);
//    f->setContentsMargins(0,1,0,1);
    fl->addRow(f2);

    auto ufl = new QLabel("FT Vertical Units");
    ufl->setAlignment(Qt::AlignCenter);
    ufl->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);
    fl->addRow(ufl);

    d_ftUnits.insert(BlackChirp::FtPlotV,QString("V"));
    d_ftUnits.insert(BlackChirp::FtPlotmV,QString("mV"));
    d_ftUnits.insert(BlackChirp::FtPlotuV,QString::fromUtf16(u"μV"));
    d_ftUnits.insert(BlackChirp::FtPlotnV,QString("nV"));

    auto unitsGroup = new QButtonGroup;
    unitsGroup->setExclusive(true);
    connect(unitsGroup,bc,this,&FtmwProcessingWidget::readSettings);

    for(auto it=d_ftUnits.constBegin(); it!=d_ftUnits.constEnd(); it++)
    {
        auto button = new QRadioButton;
        if(it.key() == BlackChirp::FtPlotuV)
            button->setChecked(true);

        unitsGroup->addButton(button);
        d_unitsButtons.insert(it.key(),button);
        auto label = new QLabel(it.value());
        label->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
        fl->addRow(label,button);

    }


    setLayout(fl);
}

void FtmwProcessingWidget::prepareForExperient(const Experiment e)
{
    setEnabled(e.ftmwConfig().isEnabled());

    if(e.ftmwConfig().isEnabled())
    {
        p_startBox->setRange(0.0,e.ftmwConfig().fidDurationUs());
        p_endBox->setRange(0.0,e.ftmwConfig().fidDurationUs());
    }

}

void FtmwProcessingWidget::applySettings(FtWorker::FidProcessingSettings s)
{
    blockSignals(true);

    p_startBox->setValue(qBound(0.0,s.startUs,s.endUs));
    p_endBox->setValue(s.endUs);
    p_autoScaleIgnoreBox->setValue(s.autoScaleIgnoreMHz);
    p_zeroPadBox->setValue(s.zeroPadFactor);
    p_removeDCBox->setChecked(s.removeDC);
    d_windowButtons[s.windowFunction]->setChecked(true);
    d_unitsButtons[s.units]->setChecked(true);

    blockSignals(false);

    readSettings();
}

void FtmwProcessingWidget::readSettings()
{
    if(signalsBlocked())
        return;

    double start = p_startBox->value();
    double stop = p_endBox->value();
    bool rdc = p_removeDCBox->isChecked();
    int zeroPad = p_zeroPadBox->value();
    double ignore = p_autoScaleIgnoreBox->value();
    auto units = BlackChirp::FtPlotuV;
    auto winf = BlackChirp::Boxcar;

    p_startBox->setMaximum(stop);
    p_endBox->setMinimum(start);


    for(auto it = d_unitsButtons.constBegin(); it != d_unitsButtons.constEnd(); it++)
    {
        if(it.value()->isChecked())
        {
            units = it.key();
            break;
        }
    }

    for(auto it = d_windowButtons.constBegin(); it != d_windowButtons.constEnd(); it++)
    {
        if(it.value()->isChecked())
        {
            winf = it.key();
            break;
        }
    }


    emit settingsUpdated(FtWorker::FidProcessingSettings{ start, stop, zeroPad, rdc, units, ignore, winf });

}

