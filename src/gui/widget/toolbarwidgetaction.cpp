#include "toolbarwidgetaction.h"

#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>
#include <QCheckBox>


QWidget *ToolBarWidgetAction::createWidget(QWidget *parent)
{
    auto out = new QWidget(parent);
    auto lbl = new QLabel(d_label,parent);
    lbl->setSizePolicy(QSizePolicy::Preferred,QSizePolicy::MinimumExpanding);

    auto fl = new QFormLayout;
    fl->setFormAlignment(Qt::AlignRight|Qt::AlignVCenter);
    fl->setContentsMargins(6,6,6,6);
    out->setToolTip(toolTip());
    p_widget = _createWidget(parent);
    fl->addRow(lbl,p_widget);
    connect(p_widget,&QWidget::destroyed,[this](){ p_widget = nullptr; });
    out->setLayout(fl);
    return out;
}

void SpinBoxWidgetAction::setSpecialValueText(const QString text)
{
    d_specialText = text;
    auto box = dynamic_cast<QSpinBox*>(p_widget);
    if(box)
        box->setSpecialValueText(text);

}

void SpinBoxWidgetAction::setRange(int min, int max)
{
    d_range = {min,max};
    auto box = dynamic_cast<QSpinBox*>(p_widget);
    if(box)
        box->setRange(min,max);
}

void SpinBoxWidgetAction::setMinimum(int min)
{
    d_range.first = min;
    auto box = dynamic_cast<QSpinBox*>(p_widget);
    if(box)
        box->setMinimum(min);
}

void SpinBoxWidgetAction::setMaximum(int max)
{
    d_range.second = max;
    auto box = dynamic_cast<QSpinBox*>(p_widget);
    if(box)
        box->setMaximum(max);
}

void SpinBoxWidgetAction::setPrefix(const QString p)
{
    d_prefix = p;
    auto box = dynamic_cast<QSpinBox*>(p_widget);
    if(box)
        box->setPrefix(p);
}

void SpinBoxWidgetAction::setSuffix(const QString s)
{
    d_suffix = s;
    auto box = dynamic_cast<QSpinBox*>(p_widget);
    if(box)
        box->setSuffix(s);
}

void SpinBoxWidgetAction::setSingleStep(int step)
{
    d_step = step;
    auto box = dynamic_cast<QSpinBox*>(p_widget);
    if(box)
        box->setSingleStep(d_step);
}

int SpinBoxWidgetAction::value() const
{
    auto box = dynamic_cast<QSpinBox*>(p_widget);
    if(box)
        return box->value();

    return d_value;
}

QWidget *SpinBoxWidgetAction::_createWidget(QWidget *parent)
{
    auto out = new QSpinBox(parent);
    out->setRange(d_range.first,d_range.second);
    out->setSpecialValueText(d_specialText);
    out->setPrefix(d_prefix);
    out->setSuffix(d_suffix);
    out->setKeyboardTracking(false);
    out->setSingleStep(d_step);
    out->setValue(d_value);

    connect(out,qOverload<int>(&QSpinBox::valueChanged),this,&SpinBoxWidgetAction::setValue);

    return out;

}

void SpinBoxWidgetAction::setValue(int v)
{
    d_value = v;
    auto box = dynamic_cast<QSpinBox*>(p_widget);
    if(box)
    {
        box->blockSignals(true);
        box->setValue(v);
        box->blockSignals(false);
    }
    emit valueChanged(d_value);
}

void DoubleSpinBoxWidgetAction::setSpecialValueText(QString text)
{
    d_specialText = text;
    auto box = dynamic_cast<QDoubleSpinBox*>(p_widget);
    if(box)
        box->setSpecialValueText(text);
}

void DoubleSpinBoxWidgetAction::setRange(double min, double max)
{
    d_range = {min,max};
    auto box = dynamic_cast<QDoubleSpinBox*>(p_widget);
    if(box)
        box->setRange(min,max);
}

void DoubleSpinBoxWidgetAction::setMinimum(double min)
{
    d_range.first = min;
    auto box = dynamic_cast<QDoubleSpinBox*>(p_widget);
    if(box)
        box->setMinimum(min);
}

void DoubleSpinBoxWidgetAction::setMaximum(double max)
{
    d_range.second = max;
    auto box = dynamic_cast<QDoubleSpinBox*>(p_widget);
    if(box)
        box->setMaximum(max);
}

void DoubleSpinBoxWidgetAction::setPrefix(const QString p)
{
    d_prefix = p;
    auto box = dynamic_cast<QDoubleSpinBox*>(p_widget);
    if(box)
        box->setPrefix(p);
}

void DoubleSpinBoxWidgetAction::setSuffix(const QString s)
{
    d_suffix = s;
    auto box = dynamic_cast<QDoubleSpinBox*>(p_widget);
    if(box)
        box->setSuffix(s);
}

void DoubleSpinBoxWidgetAction::setDecimals(int d)
{
    d_decimals = qBound(0,d,15);
    auto box = dynamic_cast<QDoubleSpinBox*>(p_widget);
    if(box)
        box->setDecimals(d_decimals);
}

void DoubleSpinBoxWidgetAction::setSingleStep(double s)
{
    d_step = s;
    auto box = dynamic_cast<QDoubleSpinBox*>(p_widget);
    if(box)
        box->setSingleStep(s);
}

double DoubleSpinBoxWidgetAction::value() const
{
    auto box = dynamic_cast<QDoubleSpinBox*>(p_widget);
    if(box)
        return box->value();

    return d_value;
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

QWidget *CheckWidgetAction::_createWidget(QWidget *parent)
{
    auto out = new QCheckBox(parent);
    out->setChecked(isChecked());
    connect(out,&QCheckBox::toggled,this,&CheckWidgetAction::toggle);

    return out;
}
