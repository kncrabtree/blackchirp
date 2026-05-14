#include "lifprocessingwidget.h"

#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QGroupBox>
#include <QLabel>
#include <QFormLayout>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>

LifProcessingWidget::LifProcessingWidget(bool store, QWidget *parent)
    : QWidget{parent}, SettingsStorage(BC::Key::LifProcessing::key)
{
    using namespace BC::Key::LifProcessing;

    auto tt = QString("Gate position in units of points. Hold Ctrl to adjust in steps of 10");
    p_lgStartBox = new QSpinBox(this);
    p_lgStartBox->setToolTip(tt);
    p_lgStartBox->setRange(0,1000000000);
    p_lgStartBox->setValue(get(lgStart,0));

    p_lgEndBox = new QSpinBox(this);
    p_lgEndBox->setToolTip(tt);
    p_lgEndBox->setRange(1,1000000000);
    p_lgEndBox->setValue(get(lgEnd,1));

    p_rgStartBox = new QSpinBox(this);
    p_rgStartBox->setToolTip(tt);
    p_rgStartBox->setRange(0,1000000000);
    p_rgStartBox->setValue(get(rgStart,0));

    p_rgEndBox = new QSpinBox(this);
    p_rgEndBox->setToolTip(tt);
    p_rgEndBox->setRange(1,1000000000);
    p_rgEndBox->setValue(get(rgEnd,1));

    auto gateBox = new QGroupBox("Gates",this);
    auto gateGrid = new QGridLayout;
    auto startHdr = new QLabel("Start",this);
    auto endHdr = new QLabel("End",this);
    startHdr->setAlignment(Qt::AlignCenter);
    endHdr->setAlignment(Qt::AlignCenter);
    gateGrid->addWidget(startHdr,0,1);
    gateGrid->addWidget(endHdr,0,2);
    gateGrid->addWidget(new QLabel("LIF",this),1,0,Qt::AlignRight);
    gateGrid->addWidget(p_lgStartBox,1,1);
    gateGrid->addWidget(p_lgEndBox,1,2);
    gateGrid->addWidget(new QLabel("Reference",this),2,0,Qt::AlignRight);
    gateGrid->addWidget(p_rgStartBox,2,1);
    gateGrid->addWidget(p_rgEndBox,2,2);
    gateBox->setLayout(gateGrid);

    p_lpAlphaBox = new QDoubleSpinBox(this);
    p_lpAlphaBox->setDecimals(4);
    p_lpAlphaBox->setRange(0.0,0.9999);
    p_lpAlphaBox->setSingleStep(0.01);
    p_lpAlphaBox->setSpecialValueText(QString("Disabled"));
    p_lpAlphaBox->setToolTip("Low pass filter: x_n = alpha*x_{n-1} + (1-alpha)*x_n");
    p_lpAlphaBox->setValue(get(lpAlpha,0.0));

    auto lpForm = new QFormLayout;
    lpForm->addRow("Low pass α",p_lpAlphaBox);

    p_sgGroupBox = new QGroupBox("Savitzky-Golay smoothing",this);
    p_sgGroupBox->setCheckable(true);
    p_sgGroupBox->setToolTip("Enable/disable Savitzky-Golay smoothing");

    p_sgWinBox = new QSpinBox(this);
    p_sgWinBox->setToolTip("Savitzky-Golay window size. Must be odd");
    p_sgWinBox->setMinimum(3);
    p_sgWinBox->setSingleStep(2);

    p_sgPolyBox = new QSpinBox(this);
    p_sgPolyBox->setToolTip("Savitzky-Golay polynomial order. Must be between 2 and window size - 1");
    p_sgPolyBox->setMinimum(2);

    auto sgHbl = new QHBoxLayout;
    sgHbl->addWidget(new QLabel("Window",this));
    sgHbl->addWidget(p_sgWinBox,1);
    sgHbl->addSpacing(8);
    sgHbl->addWidget(new QLabel("Order",this));
    sgHbl->addWidget(p_sgPolyBox,1);
    p_sgGroupBox->setLayout(sgHbl);

    p_reprocessButton = new QPushButton(QString("Reprocess All"),this);
    p_reprocessButton->setEnabled(false);
    connect(p_reprocessButton,&QPushButton::clicked,this,&LifProcessingWidget::reprocessSignal);

    p_resetButton = new QPushButton(QString("Reset"),this);
    p_resetButton->setToolTip("Reset to most recently saved values");
    p_resetButton->setEnabled(false);
    connect(p_resetButton,&QPushButton::clicked,this,&LifProcessingWidget::resetSignal);

    p_saveButton = new QPushButton(QString("Save"),this);
    p_saveButton->setToolTip("Save the current values. They will be the new defaults if this experiment is viewed again.");
    p_saveButton->setEnabled(false);
    connect(p_saveButton,&QPushButton::clicked,this,&LifProcessingWidget::saveSignal);

    auto btnHbl = new QHBoxLayout;
    btnHbl->addWidget(p_reprocessButton,1);
    btnHbl->addWidget(p_resetButton,1);
    btnHbl->addWidget(p_saveButton,1);

    auto vbl = new QVBoxLayout;
    vbl->addWidget(gateBox);
    vbl->addLayout(lpForm);
    vbl->addWidget(p_sgGroupBox);
    vbl->addLayout(btnHbl);
    vbl->addStretch(1);
    setLayout(vbl);

    connect(p_lgStartBox,qOverload<int>(&QSpinBox::valueChanged),this,[this](int n){
        auto v = p_lgEndBox->value();
        if(n >= v)
            p_lgEndBox->setValue(n+1);
    });
    connect(p_lgEndBox,qOverload<int>(&QSpinBox::valueChanged),this,[this](int n){
        auto v = p_lgStartBox->value();
        if(v >= n)
            p_lgStartBox->setValue(n-1);
    });
    connect(p_rgStartBox,qOverload<int>(&QSpinBox::valueChanged),this,[this](int n){
        auto v = p_rgEndBox->value();
        if(n >= v)
            p_rgEndBox->setValue(n+1);
    });
    connect(p_rgEndBox,qOverload<int>(&QSpinBox::valueChanged),this,[this](int n){
        auto v = p_rgStartBox->value();
        if(v >= n)
            p_rgStartBox->setValue(n-1);
    });
    connect(p_sgWinBox,qOverload<int>(&QSpinBox::valueChanged),this,[this](int n){
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
    connect(p_sgGroupBox,&QGroupBox::toggled,this,&LifProcessingWidget::settingChanged);
    connect(p_sgWinBox,qOverload<int>(&QSpinBox::valueChanged),this,&LifProcessingWidget::settingChanged);
    connect(p_sgPolyBox,qOverload<int>(&QSpinBox::valueChanged),this,&LifProcessingWidget::settingChanged);

    p_sgGroupBox->setChecked(get(sgEn,false));
    p_sgWinBox->setValue(get(sgWin,11));
    p_sgPolyBox->setValue(get(sgPoly,3));

    if(store)
    {
        registerGetter(lgStart,p_lgStartBox,&QSpinBox::value);
        registerGetter(lgEnd,p_lgEndBox,&QSpinBox::value);
        registerGetter(rgStart,p_rgStartBox,&QSpinBox::value);
        registerGetter(rgEnd,p_rgEndBox,&QSpinBox::value);
        registerGetter(lpAlpha,p_lpAlphaBox,&QDoubleSpinBox::value);
        registerGetter(sgEn,p_sgGroupBox,&QGroupBox::isChecked);
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
    p_sgGroupBox->setChecked(lc.savGolEnabled);
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
                p_sgGroupBox->isChecked(),
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
