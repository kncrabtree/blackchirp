#include "clockdisplaybox.h"

#include <QGridLayout>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QMetaEnum>

#include <data/experiment/rfconfig.h>

ClockDisplayBox::ClockDisplayBox(QWidget *parent) : QGroupBox(parent)
{
    setTitle("Clocks");
    auto gl = new QGridLayout;
    gl->setSpacing(3);
    gl->setContentsMargins(3,3,3,3);

    auto ct = QMetaEnum::fromType<RfConfig::ClockType>();

    for(int i=0; i<ct.keyCount(); i++)
    {
        auto key = ct.key(i);

        auto *box = new QDoubleSpinBox(this);
        box->setRange(-1.0,1e7);
        box->setDecimals(6);
        box->setSuffix(QString(" MHz"));
        box->setButtonSymbols(QAbstractSpinBox::NoButtons);
        box->setReadOnly(true);
        box->setSpecialValueText(QString("Not Configured"));
        box->setValue(-1.0);
        box->blockSignals(true);

        auto *lbl = new QLabel(key);
        lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);

        gl->addWidget(lbl,i,0);
        gl->addWidget(box,i,1);
        d_boxes.insert(static_cast<RfConfig::ClockType>(ct.value(i)),box);
    }
    gl->setColumnStretch(0,0);
    gl->setColumnStretch(1,1);
    setLayout(gl);
    setSizePolicy(QSizePolicy::Maximum,QSizePolicy::Maximum);
}

void ClockDisplayBox::updateFrequency(RfConfig::ClockType t, double f)
{
    auto box = d_boxes.value(t);
    box->setValue(f);
    box->setSpecialValueText(QString("Error"));
}


QSize ClockDisplayBox::sizeHint() const
{
    return QGroupBox::sizeHint();
}
