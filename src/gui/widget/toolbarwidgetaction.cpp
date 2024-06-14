#include "toolbarwidgetaction.h"

#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QSignalBlocker>

QWidget *ToolBarWidgetAction::createWidget(QWidget *parent)
{
    auto out = new QWidget(parent);
    auto lbl = new QLabel(d_label,out);
    lbl->setSizePolicy(QSizePolicy::Preferred,QSizePolicy::MinimumExpanding);

    auto fl = new QFormLayout;
    fl->setFormAlignment(Qt::AlignHCenter|Qt::AlignVCenter);
    fl->setContentsMargins(6,6,6,6);
    out->setToolTip(toolTip());
    p_widget = _createWidget(out);
    p_widget->setObjectName("ActionWidget");
    fl->addRow(lbl,p_widget);
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
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QSpinBox*>("ActionWidget");
        if(box)
            box->setRange(min,max);
    }
}

void SpinBoxWidgetAction::setMinimum(int min)
{
    d_range.first = min;
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QSpinBox*>("ActionWidget");
        if(box)
            box->setMinimum(min);
    }
}

void SpinBoxWidgetAction::setMaximum(int max)
{
    d_range.second = max;
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QSpinBox*>("ActionWidget");
        if(box)
            box->setMaximum(max);
    }
}

void SpinBoxWidgetAction::setPrefix(const QString p)
{
    d_prefix = p;
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QSpinBox*>("ActionWidget");
        if(box)
            box->setPrefix(p);
    }
}

void SpinBoxWidgetAction::setSuffix(const QString s)
{
    d_suffix = s;
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QSpinBox*>("ActionWidget");
        if(box)
            box->setSuffix(s);
    }
}

void SpinBoxWidgetAction::setSingleStep(int step)
{
    d_step = step;
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QSpinBox*>("ActionWidget");
        if(box)
            box->setSingleStep(d_step);
    }
}

int SpinBoxWidgetAction::value() const
{
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QSpinBox*>("ActionWidget");
        if(box)
            return box->value();
    }

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
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QSpinBox*>("ActionWidget");
        if(box)
        {
            box->blockSignals(true);
            box->setValue(v);
            box->blockSignals(false);
        }
    }

    emit valueChanged(d_value);
}

void DoubleSpinBoxWidgetAction::setSpecialValueText(QString text)
{
    d_specialText = text;
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QDoubleSpinBox*>("ActionWidget");
        if(box)
            box->setSpecialValueText(text);
    }
}

void DoubleSpinBoxWidgetAction::setRange(double min, double max)
{
    d_range = {min,max};
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QDoubleSpinBox*>("ActionWidget");
        if(box)
            box->setRange(min,max);
    }
}

void DoubleSpinBoxWidgetAction::setMinimum(double min)
{
    d_range.first = min;
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QDoubleSpinBox*>("ActionWidget");
        if(box)
            box->setMinimum(min);
    }
}

void DoubleSpinBoxWidgetAction::setMaximum(double max)
{
    d_range.second = max;
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QDoubleSpinBox*>("ActionWidget");
        if(box)
            box->setMaximum(max);
    }
}

void DoubleSpinBoxWidgetAction::setPrefix(const QString p)
{
    d_prefix = p;
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QDoubleSpinBox*>("ActionWidget");
        if(box)
            box->setPrefix(p);
    }
}

void DoubleSpinBoxWidgetAction::setSuffix(const QString s)
{
    d_suffix = s;
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QDoubleSpinBox*>("ActionWidget");
        if(box)
            box->setSuffix(s);
    }
}

void DoubleSpinBoxWidgetAction::setDecimals(int d)
{
    d_decimals = qBound(0,d,15);
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QDoubleSpinBox*>("ActionWidget");
        if(box)
            box->setDecimals(d_decimals);
    }
}

void DoubleSpinBoxWidgetAction::setSingleStep(double s)
{
    d_step = s;
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QDoubleSpinBox*>("ActionWidget");
        if(box)
            box->setSingleStep(s);
    }
}

double DoubleSpinBoxWidgetAction::value() const
{
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QDoubleSpinBox*>("ActionWidget");
        if(box)
            return box->value();
    }

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

void DoubleSpinBoxWidgetAction::setValue(double v) {
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QDoubleSpinBox*>("ActionWidget");
        if(box)
        {
            QSignalBlocker b(box);
            box->setValue(v);
        }
    }
    d_value = v;
    emit valueChanged(v);
}

QWidget *CheckWidgetAction::_createWidget(QWidget *parent)
{
    auto out = new QCheckBox(parent);
    out->setChecked(isChecked());
    connect(out,&QCheckBox::toggled,this,&CheckWidgetAction::checkStateChanged);

    return out;
}

void CheckWidgetAction::setCheckedState(bool b)
{
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QCheckBox*>("ActionWidget");
        if(box)
        {
            QSignalBlocker bl(box);
            box->setChecked(b);
        }
    }

    d_checked = b;
    emit checkStateChanged(b);
}

bool CheckWidgetAction::readCheckedState() const
{
    for(auto w : createdWidgets())
    {
        auto box = w->findChild<QCheckBox*>("ActionWidget");
        if(box)
            return box->isChecked();
    }

    return d_checked;
}
