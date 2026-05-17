#include "gascontrolwidget.h"

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QTableWidget>
#include <QVBoxLayout>

#include <hardware/optional/flowcontroller/flowcontroller.h>
#include <gui/widget/cellwidgethelpers.h>

using BC::Gui::centerCellWidget;

namespace {
constexpr int LabelCol = 0;
constexpr int NameCol = 1;
constexpr int SetpointCol = 2;
constexpr int EnabledCol = 3;
}

GasControlWidget::GasControlWidget(const FlowConfig &cfg, QWidget *parent) :
    QWidget(parent),
    SettingsStorage(BC::Key::widgetKey(BC::Key::GasControl::key,cfg.headerKey())),
    d_config{cfg}
{
    SettingsStorage fc(cfg.headerKey(),Hardware);
    auto flowChannels = fc.get(BC::Key::Flow::flowChannels,4);

    p_table = new QTableWidget(flowChannels, 4, this);
    p_table->setHorizontalHeaderLabels({"Ch", "Gas Name", "Setpoint", "Enabled"});
    p_table->horizontalHeader()->setSectionResizeMode(LabelCol, QHeaderView::ResizeToContents);
    p_table->horizontalHeader()->setSectionResizeMode(NameCol, QHeaderView::Stretch);
    p_table->horizontalHeader()->setSectionResizeMode(SetpointCol, QHeaderView::ResizeToContents);
    p_table->horizontalHeader()->setSectionResizeMode(EnabledCol, QHeaderView::ResizeToContents);
    p_table->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    p_table->verticalHeader()->setVisible(false);
    p_table->setSelectionMode(QAbstractItemView::NoSelection);
    p_table->setFocusPolicy(Qt::NoFocus);

    for(int i=0; i<flowChannels; ++i)
    {
        auto labelItem = new QTableWidgetItem(QString::number(i+1));
        labelItem->setFlags(Qt::ItemIsEnabled);
        labelItem->setTextAlignment(Qt::AlignCenter);
        p_table->setItem(i, LabelCol, labelItem);

        auto nameItem = new QTableWidgetItem(
            fc.getArrayValue(BC::Key::Flow::channels,i,BC::Key::Flow::chName,QString("")));
        p_table->setItem(i, NameCol, nameItem);

        auto controlBox = new QDoubleSpinBox;
        controlBox->setSpecialValueText(QString("Off"));
        controlBox->setKeyboardTracking(false);
        connect(controlBox, qOverload<double>(&QDoubleSpinBox::valueChanged),
                this, [this,i](double v) { emit gasSetpointUpdate(d_config.headerKey(),i,v); });
        p_table->setCellWidget(i, SetpointCol, controlBox);

        auto enableBox = new QCheckBox;
        connect(enableBox, &QCheckBox::toggled, this, [this,i](bool en){
            emit enableUpdate(d_config.headerKey(), i, en);
        });
        centerCellWidget(p_table, i, EnabledCol, enableBox);

        d_widgets.append({controlBox, enableBox});
    }

    connect(p_table, &QTableWidget::itemChanged, this, [this](QTableWidgetItem *item){
        if(!item || item->column() != NameCol)
            return;
        emit nameUpdate(d_config.headerKey(), item->row(), item->text());
    });

    auto channelsBox = new QGroupBox(QString("Flow Channels"), this);
    auto channelsLayout = new QVBoxLayout(channelsBox);
    channelsLayout->setContentsMargins(3,3,3,3);
    channelsLayout->addWidget(p_table);

    auto pressureBox = new QGroupBox(QString("Pressure"), this);
    auto pressureLayout = new QHBoxLayout(pressureBox);

    auto setpointLabel = new QLabel(QString("Setpoint:"), pressureBox);
    setpointLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

    p_pressureSetpointBox = new QDoubleSpinBox(pressureBox);
    connect(p_pressureSetpointBox, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double sp){
        emit pressureSetpointUpdate(d_config.headerKey(),sp);
    });

    p_pressureControlBox = new QCheckBox(pressureBox);
    p_pressureControlBox->setToolTip(QString("Enable automatic pressure feedback "
                                             "control. The controller adjusts flow "
                                             "rates to maintain the setpoint."));
    connect(p_pressureControlBox, &QCheckBox::toggled, this, [this](bool en){
        emit pressureControlUpdate(d_config.headerKey(),en);
    });

    auto controlLabel = new QLabel(QString("Auto Control"), pressureBox);
    controlLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);

    pressureLayout->addWidget(setpointLabel, 1);
    pressureLayout->addWidget(p_pressureSetpointBox);
    pressureLayout->addWidget(p_pressureControlBox);
    pressureLayout->addWidget(controlLabel, 1);

    auto outerVbl = new QVBoxLayout(this);
    outerVbl->setContentsMargins(0,0,0,0);
    outerVbl->addWidget(channelsBox, 1);
    outerVbl->addWidget(pressureBox, 0);

    applySettings();
    initialize(d_config);
}

FlowConfig &GasControlWidget::toConfig()
{
    d_config.d_pressureControlMode = p_pressureControlBox->isChecked();
    d_config.d_pressureSetpoint = p_pressureSetpointBox->value();
    for(int i=0; i<d_widgets.size(); i++)
    {
        const auto &w = d_widgets.at(i);
        d_config.setCh(i, FlowConfig::Setpoint, w.setpointBox->value());
        d_config.setCh(i, FlowConfig::Enabled, w.enableBox->isChecked());
        if(auto item = p_table->item(i, NameCol))
            d_config.setCh(i, FlowConfig::Name, item->text());
    }

    return d_config;
}

void GasControlWidget::initialize(const FlowConfig &cfg)
{
    if(cfg.headerKey() != d_config.headerKey())
        return;

    for(int i=0; i<cfg.size(); ++i)
    {
        updateGasSetpoint(cfg.headerKey(), i, cfg.setting(i,FlowConfig::Setpoint).toDouble());
        updateChannelEnabled(cfg.headerKey(), i, cfg.setting(i,FlowConfig::Enabled).toBool());
    }

    updatePressureSetpoint(cfg.headerKey(), cfg.d_pressureSetpoint);
    updatePressureControl(cfg.headerKey(), cfg.d_pressureControlMode);
}

void GasControlWidget::applySettings()
{
    using namespace BC::Key::Flow;
    SettingsStorage fc(d_config.headerKey(),Hardware);

    p_pressureSetpointBox->setDecimals(fc.get(pDec,3));
    p_pressureSetpointBox->setMaximum(fc.get(pMax,10.0));
    p_pressureSetpointBox->setSuffix(QString(" ") + fc.get(pUnits,QString("")));

    for(int i=0; i<d_widgets.size(); ++i)
    {
        auto b = d_widgets.at(i).setpointBox;
        b->setDecimals(fc.getArrayValue(channels,i,chDecimals,2));
        b->setMaximum(fc.getArrayValue(channels,i,chMax,10000.0));
        b->setSuffix(QString(" ")+fc.getArrayValue(channels,i,chUnits,QString("")));
    }
}

void GasControlWidget::updateGasSetpoint(const QString key, int i, double sp)
{
    if(key != d_config.headerKey())
        return;

    if(i < 0 || i >= d_widgets.size())
        return;

    auto b = d_widgets.at(i).setpointBox;
    if(!b->hasFocus())
    {
        b->blockSignals(true);
        b->setValue(sp);
        b->blockSignals(false);
    }
}

void GasControlWidget::updateChannelEnabled(const QString key, int i, bool en)
{
    if(key != d_config.headerKey())
        return;

    if(i < 0 || i >= d_widgets.size())
        return;

    auto cb = d_widgets.at(i).enableBox;
    if(cb->hasFocus())
        return;

    cb->blockSignals(true);
    cb->setChecked(en);
    cb->blockSignals(false);
}

void GasControlWidget::updatePressureSetpoint(const QString key, double sp)
{
    if(key != d_config.headerKey())
        return;

    if(!p_pressureSetpointBox->hasFocus())
    {
        p_pressureSetpointBox->blockSignals(true);
        p_pressureSetpointBox->setValue(sp);
        p_pressureSetpointBox->blockSignals(false);
    }
}

void GasControlWidget::updatePressureControl(const QString key, bool en)
{
    if(key != d_config.headerKey())
        return;

    if(p_pressureControlBox->hasFocus())
        return;

    p_pressureControlBox->blockSignals(true);
    p_pressureControlBox->setChecked(en);
    p_pressureControlBox->blockSignals(false);
}
