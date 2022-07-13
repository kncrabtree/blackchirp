#include "lifprocessingwidget.h"

#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QCheckBox>
#include <QFormLayout>

LifProcessingWidget::LifProcessingWidget(bool store, QWidget *parent)
    : QWidget{parent}, SettingsStorage(BC::Key::LifProcessing::key)
{
    using namespace BC::Key::LifProcessing;
    auto fl = new QFormLayout(this);

    auto tt = QString("Gate position in units of points. Hold Ctrl to adjust in steps of 10");
    p_lgStartBox = new QSpinBox(this);
    p_lgStartBox->setToolTip(tt);
    p_lgStartBox->setRange(0,1000000000);
    p_lgStartBox->setValue(get(lgStart,0));
    fl->addRow("LIF Gate Start",p_lgStartBox);

    p_lgEndBox = new QSpinBox(this);
    p_lgEndBox->setToolTip(tt);
    p_lgEndBox->setRange(1,1000000000);
    p_lgEndBox->setValue(get(lgEnd,1));
    fl->addRow("LIF Gate End",p_lgEndBox);

    p_rgStartBox = new QSpinBox(this);
    p_rgStartBox->setToolTip(tt);
    p_rgStartBox->setRange(0,1000000000);
    p_rgStartBox->setValue(get(rgStart,0));
    fl->addRow("Reference Gate Start",p_rgStartBox);

    p_rgEndBox = new QSpinBox(this);
    p_rgEndBox->setToolTip(tt);
    p_rgEndBox->setRange(1,1000000000);
    p_rgEndBox->setValue(get(rgEnd,1));
    fl->addRow("Reference Gate End",p_rgEndBox);

    p_lpAlphaBox = new QDoubleSpinBox(this);
    p_lpAlphaBox->setDecimals(4);
    p_lpAlphaBox->setRange(0.0,0.9999);
    p_lpAlphaBox->setSingleStep(0.01);
    p_lpAlphaBox->setSpecialValueText(QString("Disabled"));
    p_lpAlphaBox->setToolTip("Low pass filter: x_n = alpha*x_{n-1} + (1-alpha)*x_n");
    p_lpAlphaBox->setValue(get(lpAlpha,0.0));
    fl->addRow("Low Pass Filter Alpha",p_lpAlphaBox);

    p_sgEnBox = new QCheckBox(this);
    p_sgEnBox->setToolTip("Enable/disable Savitsky-Golay smoothing");
    fl->addRow("Savitzky-Golay Filter Enabled",p_sgEnBox);

    p_sgWinBox = new QSpinBox(this);
    p_sgWinBox->setToolTip("Savitzky-Golay window size. Must be odd");
    p_sgWinBox->setMinimum(3);
    p_sgWinBox->setSingleStep(2);
    p_sgWinBox->setEnabled(false);
    fl->addRow("Savitzky-Golay Window",p_sgWinBox);

    p_sgPolyBox = new QSpinBox(this);
    p_sgPolyBox->setToolTip("Savitzky-Golay polynomial order. Must be between 2 and window size - 1");
    p_sgPolyBox->setMinimum(2);
    p_sgPolyBox->setEnabled(false);
    fl->addRow("Savitzky-Golay Polynomial Order",p_sgPolyBox);

    connect(p_sgEnBox,&QCheckBox::toggled,p_sgWinBox,&QSpinBox::setEnabled);
    connect(p_sgEnBox,&QCheckBox::toggled,p_sgPolyBox,&QSpinBox::setEnabled);
    connect(p_lgStartBox,qOverload<int>(&QSpinBox::valueChanged),[=](int n){
        auto v = p_lgEndBox->value();
        if(n >= v)
            p_lgEndBox->setValue(n+1);
    });
    connect(p_lgEndBox,qOverload<int>(&QSpinBox::valueChanged),[=](int n){
        auto v = p_lgStartBox->value();
        if(v >= n)
            p_lgStartBox->setValue(n-1);
    });
    connect(p_rgStartBox,qOverload<int>(&QSpinBox::valueChanged),[=](int n){
        auto v = p_rgEndBox->value();
        if(n >= v)
            p_rgEndBox->setValue(n+1);
    });
    connect(p_rgEndBox,qOverload<int>(&QSpinBox::valueChanged),[=](int n){
        auto v = p_rgStartBox->value();
        if(v >= n)
            p_rgStartBox->setValue(n-1);
    });
    connect(p_sgWinBox,qOverload<int>(&QSpinBox::valueChanged),[=](int n){
        if(!(n%2))
        {
            n--;
            p_sgWinBox->blockSignals(true);
            p_sgWinBox->setValue(n);
            p_sgWinBox->blockSignals(false);
        }

        p_sgPolyBox->setMaximum(n-1);
    });

    connect(p_lgStartBox,qOverload<int>(&QSpinBox::valueChanged),this,&LifProcessingWidget::settingChanged);
    connect(p_lgEndBox,qOverload<int>(&QSpinBox::valueChanged),this,&LifProcessingWidget::settingChanged);
    connect(p_rgStartBox,qOverload<int>(&QSpinBox::valueChanged),this,&LifProcessingWidget::settingChanged);
    connect(p_rgEndBox,qOverload<int>(&QSpinBox::valueChanged),this,&LifProcessingWidget::settingChanged);
    connect(p_lpAlphaBox,qOverload<double>(&QDoubleSpinBox::valueChanged),this,&LifProcessingWidget::settingChanged);
    connect(p_sgEnBox,&QAbstractButton::toggled,this,&LifProcessingWidget::settingChanged);
    connect(p_sgWinBox,qOverload<int>(&QSpinBox::valueChanged),this,&LifProcessingWidget::settingChanged);
    connect(p_sgPolyBox,qOverload<int>(&QSpinBox::valueChanged),this,&LifProcessingWidget::settingChanged);


    p_sgEnBox->setChecked(get(sgEn,false));
    p_sgWinBox->setValue(get(sgWin,11));
    p_sgPolyBox->setValue(get(sgPoly,3));

    p_reprocessButton = new QPushButton(QString("Reprocess All"),this);
    p_reprocessButton->setEnabled(false);
    connect(p_reprocessButton,&QPushButton::clicked,this,&LifProcessingWidget::reprocessSignal);

    fl->addRow("",p_reprocessButton);

    p_resetButton = new QPushButton(QString("Reset"),this);
    p_resetButton->setToolTip("Reset to most recently saved values");
    p_resetButton->setEnabled(false);
    connect(p_resetButton,&QPushButton::clicked,this,&LifProcessingWidget::resetSignal);
    fl->addRow("",p_resetButton);

    p_saveButton = new QPushButton(QString("Save"),this);
    p_saveButton->setToolTip("Save the current values. They will be the new defaults if this experiment is viewed again.");
    p_saveButton->setEnabled(false);
    connect(p_saveButton,&QPushButton::clicked,this,&LifProcessingWidget::saveSignal);
    fl->addRow("",p_saveButton);

    setLayout(fl);

    if(store)
    {
        registerGetter(lgStart,p_lgStartBox,&QSpinBox::value);
        registerGetter(lgEnd,p_lgEndBox,&QSpinBox::value);
        registerGetter(rgStart,p_rgStartBox,&QSpinBox::value);
        registerGetter(rgEnd,p_rgEndBox,&QSpinBox::value);
        registerGetter(lpAlpha,p_lpAlphaBox,&QDoubleSpinBox::value);
        registerGetter(sgEn,static_cast<QAbstractButton*>(p_sgEnBox),&QAbstractButton::isChecked);
        registerGetter(sgWin,p_sgWinBox,&QSpinBox::value);
        registerGetter(sgPoly,p_sgPolyBox,&QSpinBox::value);
    }
}

void LifProcessingWidget::initialize(int recLen, bool ref)
{
    p_lgStartBox->setRange(0,recLen-2);
    p_lgEndBox->setRange(1,recLen-1);
    p_rgStartBox->setRange(0,recLen-2);
    p_rgEndBox->setRange(1,recLen-1);

    p_rgStartBox->setEnabled(ref);
    p_rgEndBox->setEnabled(ref);
}

void LifProcessingWidget::setAll(const LifTrace::LifProcSettings &lc)
{
    blockSignals(true);
    p_lgStartBox->setValue(lc.lifGateStart);
    p_lgEndBox->setValue(lc.lifGateEnd);
    p_rgStartBox->setValue(lc.refGateStart);
    p_rgEndBox->setValue(lc.refGateEnd);
    p_lpAlphaBox->setValue(lc.lowPassAlpha);
    p_sgEnBox->setChecked(lc.savGolEnabled);
    p_sgWinBox->setValue(lc.savGolWin);
    p_sgPolyBox->setValue(lc.savGolPoly);
    blockSignals(false);
    emit settingChanged();
}

LifTrace::LifProcSettings LifProcessingWidget::getSettings() const
{
    return {p_lgStartBox->value(),
                p_lgEndBox->value(),
                p_rgStartBox->value(),
                p_rgEndBox->value(),
                p_lpAlphaBox->value(),
                p_sgEnBox->isChecked(),
                p_sgWinBox->value(),
                p_sgPolyBox->value()
    };
}

void LifProcessingWidget::experimentComplete()
{
    setEnabled(true);
    p_reprocessButton->setEnabled(true);
    p_resetButton->setEnabled(true);
    p_saveButton->setEnabled(true);
}
