#include "clockdisplaywidget.h"

#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QMetaEnum>

#include <data/experiment/rfconfig.h>

ClockDisplayWidget::ClockDisplayWidget(QWidget *parent) : QWidget(parent)
{
    auto *fl = new QFormLayout();
    fl->setMargin(3);
    fl->setContentsMargins(3,3,3,3);
    fl->setSpacing(3);

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
        box->setSpecialValueText(QString("Not Yet Set"));
        box->setValue(-1.0);
        box->blockSignals(true);

        auto *lbl = new QLabel(key);
        lbl->setAlignment(Qt::AlignRight|Qt::AlignCenter);
        lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);

        fl->addRow(lbl,box);
        d_boxes.insert(static_cast<RfConfig::ClockType>(ct.value(i)),box);
    }

    setLayout(fl);
}

void ClockDisplayWidget::updateFrequency(RfConfig::ClockType t, double f)
{
    auto box = d_boxes.value(t);
    box->setValue(f);
    box->setSpecialValueText(QString("Error"));
}
