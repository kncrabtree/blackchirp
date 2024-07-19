#include "liflaserwidget.h"

#include <QPushButton>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QLabel>

#include <data/bcglobals.h>
#include <data/storage/settingsstorage.h>
#include <modules/lif/hardware/liflaser/liflaser.h>

LifLaserWidget::LifLaserWidget(QWidget *parent)
    : QWidget{parent}
{

    using namespace BC::Key::LifLaser;
    auto gl = new QGridLayout;

    SettingsStorage s(BC::Key::hwKey(key,0),SettingsStorage::Hardware);

    p_posBox = new QDoubleSpinBox;
    p_posBox->setMinimum(s.get(minPos,200.0));
    p_posBox->setMaximum(s.get(maxPos,2000.0));
    p_posBox->setSuffix(QString(" ").append(s.get(units,"nm").toString()));
    p_posBox->setDecimals(s.get(decimals,2));

    p_posSetButton = new QPushButton(QString("Set"));
    connect(p_posSetButton,&QPushButton::clicked,this,[this](){
        p_posBox->setEnabled(false);
        p_posSetButton->setEnabled(false);
        emit changePosition(p_posBox->value());
    });

    gl->addWidget(p_posBox,0,0);
    gl->addWidget(p_posSetButton,0,1);

    if(s.get(hasFl,true))
    {
        auto fl = new QLabel("Flashlamp");
        fl->setAlignment(Qt::AlignRight);
        gl->addWidget(fl,1,0);

        p_flButton = new QPushButton(QString("Enable"));
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
    if(d >= p_posBox->minimum() && d <= p_posBox->maximum())
        p_posBox->setValue(d);

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
