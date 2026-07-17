#include "liflaserwidget.h"
#include <gui/style/themecolors.h>

#include <QPushButton>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QLabel>

#include <data/bcglobals.h>
#include <data/storage/settingsstorage.h>
#include <data/storage/enumcsvconvert.h>
#include <hardware/core/liflaser/liflaser.h>
#include <hardware/optional/laserfreqconversion/laserfreqconversionstage.h>

LifLaserWidget::LifLaserWidget(const QString& lifLaserKey, QWidget *parent)
    : QWidget{parent}
{

    using namespace BC::Key::LifLaser;
    using namespace BC::LifConv;
    using namespace Qt::StringLiterals;
    auto gl = new QGridLayout;

    SettingsStorage s(lifLaserKey, SettingsStorage::Hardware);

    // minPos/maxPos are the grating fundamental's native range (cm⁻¹);
    // the box presents the FINAL-beam output range in the display unit,
    // assembled from the current LIF preset's conversion topology
    // (GUI-thread settings snapshot, no threaded device access).
    auto fundMin = s.get(minPos, 5000.0);
    auto fundMax = s.get(maxPos, 40000.0);
    d_unit = BC::CSV::enumFromVariant<LaserUnit>(s.get(units, QVariant::fromValue(LaserUnit::Nm)), LaserUnit::Nm);
    auto conv = assembleCurrentLifConversion().conversion;
    auto [outLoCm1, outHiCm1] = conv.outputRange(fundMin, fundMax);
    auto dlo = fromCm1(outLoCm1, d_unit);
    auto dhi = fromCm1(outHiCm1, d_unit);

    p_posBox = new QDoubleSpinBox;
    // A reciprocal unit (e.g. nm) reverses the min/max order.
    p_posBox->setMinimum(qMin(dlo, dhi));
    p_posBox->setMaximum(qMax(dlo, dhi));
    p_posBox->setSuffix(QString(" ").append(unitLabel(d_unit)));
    p_posBox->setDecimals(s.get(decimals,2));
    // No real position has been read yet (first update arrives via
    // setPosition() below, driven by LifLaser::laserPosUpdate); show a
    // placeholder rather than an arbitrary boundary value until then.
    p_posBox->setSpecialValueText(u"--"_s);
    p_posBox->setValue(p_posBox->minimum());

    p_posSetButton = new QPushButton(QString("Set"));
    p_posSetButton->setIcon(ThemeColors::createThemedIcon(":/icons/arrow-right-circle.svg", ThemeColors::IconPrimary, this));
    connect(p_posSetButton,&QPushButton::clicked,this,[this](){
        p_posBox->setEnabled(false);
        p_posSetButton->setEnabled(false);
        emit changePosition(BC::LifConv::toCm1(p_posBox->value(), d_unit));
    });

    gl->addWidget(p_posBox,0,0);
    gl->addWidget(p_posSetButton,0,1);

    if(s.get(hasFl,true))
    {
        auto fl = new QLabel("Flashlamp");
        fl->setAlignment(Qt::AlignRight);
        gl->addWidget(fl,1,0);

        p_flButton = new QPushButton(QString("Enable"));
        p_flButton->setIcon(ThemeColors::createThemedIcon(":/icons/light-bulb.svg", ThemeColors::IconPrimary, this));
        p_flButton->setChecked(false);
        p_flButton->setCheckable(true);
        connect(p_flButton,&QPushButton::clicked,this,[this](bool en){
            if(en)
                p_flButton->setText("Disable");
            else
                p_flButton->setText("Enable");
            p_flButton->setEnabled(false);
            emit changeFlashlamp(en);
        });
        gl->addWidget(p_flButton,1,1);
    }
    else
        p_flButton = nullptr;


    setLayout(gl);
}

void LifLaserWidget::setPosition(const double d)
{
    // d is the output-beam wavenumber (cm⁻¹); the box and its range are
    // in the display unit. d<=0 is the shared "unresolved" sentinel (see
    // LifLaser::readPos()/LifConversion::stageInput()); guard it explicitly
    // rather than relying on the min/max range check below, which only
    // rejects it by coincidence when the laser's own range excludes it.
    if(d > 0.0)
    {
        auto displayPos = BC::LifConv::fromCm1(d, d_unit);
        if(displayPos >= p_posBox->minimum() && displayPos <= p_posBox->maximum())
            p_posBox->setValue(displayPos);
    }

    p_posSetButton->setEnabled(true);
    p_posBox->setEnabled(true);
}

void LifLaserWidget::setFlashlamp(bool b)
{
    if(!p_flButton)
        return;
    p_flButton->setChecked(b);
    if(b)
        p_flButton->setText("Disable");
    else
        p_flButton->setText("Enable");
    p_flButton->setEnabled(true);
}
