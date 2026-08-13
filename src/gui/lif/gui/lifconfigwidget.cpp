#include <gui/lif/gui/lifconfigwidget.h>

#include <QTabWidget>
#include <QVBoxLayout>

#include <gui/lif/gui/lifcontrolwidget.h>
#include <gui/lif/gui/lifconversionwidget.h>

using namespace Qt::StringLiterals;

LifConfigWidget::LifConfigWidget(const QString &digitizerHwKey, const QString &laserHwKey,
                                 bool showDeleteButton, QWidget *parent)
    : QWidget(parent)
{
    p_lcw = new LifControlWidget(digitizerHwKey, laserHwKey);
    p_conversionWidget = new LifConversionWidget(showDeleteButton, this);

    p_tabWidget = new QTabWidget(this);
    p_tabWidget->addTab(p_lcw, "Acquisition"_L1);
    p_tabWidget->addTab(p_conversionWidget, "Conversion"_L1);

    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(p_tabWidget);

    connect(p_conversionWidget, &LifConversionWidget::edited, this, &LifConfigWidget::edited);
    connect(p_conversionWidget, &LifConversionWidget::dirtyChanged, this, &LifConfigWidget::dirtyChanged);
}

void LifConfigWidget::setFromConfig(const LifConfig &cfg)
{
    p_lcw->setFromConfig(cfg);
    p_conversionWidget->setFromConfig(cfg);
}

void LifConfigWidget::toConfig(LifConfig &cfg)
{
    p_lcw->toConfig(cfg);
    p_conversionWidget->toConfig(cfg);
}

LifPreset LifConfigWidget::toLifPreset() const
{
    return p_conversionWidget->toLifPreset();
}

bool LifConfigWidget::isDirty() const
{
    return p_conversionWidget->isDirty();
}

void LifConfigWidget::clearDirty()
{
    p_conversionWidget->clearDirty();
}
