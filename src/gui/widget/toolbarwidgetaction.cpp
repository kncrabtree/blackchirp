#include "toolbarwidgetaction.h"

#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>


QWidget *ToolBarWidgetAction::createWidget(QWidget *parent)
{
    auto out = new QWidget(parent);

//    if(labelText().isEmpty())
//    {
//        auto hbl = new QHBoxLayout;
//        hbl->setContentsMargins(6,6,6,6);
//        hbl->addWidget(_createWidget(parent));
//        out->setLayout(hbl);
//        return out;
//    }

    auto lbl = new QLabel(labelText(),parent);
    lbl->setSizePolicy(QSizePolicy::Preferred,QSizePolicy::MinimumExpanding);

    auto fl = new QFormLayout;
    fl->setFormAlignment(Qt::AlignRight|Qt::AlignVCenter);
    fl->setContentsMargins(6,6,6,6);
    fl->addRow(lbl,_createWidget(parent));
    out->setLayout(fl);
    return out;
}

QWidget *SpinBoxWidgetAction::_createWidget(QWidget *parent)
{
    auto out = new QSpinBox(parent);
    out->setRange(d_range.first,d_range.second);
    out->setSpecialValueText(d_specialText);
    out->setPrefix(d_prefix);
    out->setSuffix(d_suffix);
    out->setKeyboardTracking(false);
    out->setValue(d_value);

    connect(out,qOverload<int>(&QSpinBox::valueChanged),this,&SpinBoxWidgetAction::setValue);

    return out;

}

QWidget *DoubleSpinBoxWidgetAction::_createWidget(QWidget *parent)
{
    auto out = new QDoubleSpinBox(parent);
    out->setRange(d_range.first,d_range.second);
    out->setDecimals(d_decimals);
    out->setSpecialValueText(d_specialText);
    out->setPrefix(d_prefix);
    out->setSuffix(d_suffix);
    out->setSingleStep(d_step);
    out->setKeyboardTracking(false);
    out->setValue(d_value);

    connect(out,qOverload<double>(&QDoubleSpinBox::valueChanged),this,&DoubleSpinBoxWidgetAction::setValue);

    return out;
}
