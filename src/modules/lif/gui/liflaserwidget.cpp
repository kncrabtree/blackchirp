#include "liflaserwidget.h"

#include <QPushButton>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QLabel>

#include <data/storage/settingsstorage.h>
#include <modules/lif/hardware/liflaser/liflaser.h>

LifLaserWidget::LifLaserWidget(QWidget *parent)
    : QWidget{parent}
{

    using namespace BC::Key::LifLaser;
    auto gl = new QGridLayout;

    SettingsStorage s(key,SettingsStorage::Hardware);

    p_posBox = new QDoubleSpinBox;
    p_posBox->setMinimum(s.get(minPos,200.0));
    p_posBox->setMaximum(s.get(maxPos,2000.0));
    p_posBox->setSuffix(QString(" ").append(s.get(units,"nm").toString()));
    p_posBox->setDecimals(s.get(decimals,2));

    p_posSetButton = new QPushButton(QString("Set"));

    gl->addWidget(p_posBox,0,0);
    gl->addWidget(p_posSetButton,0,1);

    auto fl = new QLabel("Flashlamp");
    fl->setAlignment(Qt::AlignRight);
    gl->addWidget(fl,1,0);

    p_flButton = new QPushButton(QString("Enable"));
    p_flButton->setCheckable(true);
    p_flButton->setChecked(false);
    gl->addWidget(p_flButton,1,1);


    setLayout(gl);
}
