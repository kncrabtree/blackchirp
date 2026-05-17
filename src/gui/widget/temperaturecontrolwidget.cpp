#include "temperaturecontrolwidget.h"

#include <hardware/optional/tempcontroller/temperaturecontroller.h>
#include <gui/widget/cellwidgethelpers.h>

#include <QCheckBox>
#include <QHeaderView>
#include <QTableWidget>
#include <QVBoxLayout>

using BC::Gui::centerCellWidget;

namespace {
constexpr int LabelCol = 0;
constexpr int NameCol = 1;
constexpr int EnabledCol = 2;
}

TemperatureControlWidget::TemperatureControlWidget(const TemperatureControllerConfig &cfg, QWidget *parent) :
    QWidget(parent),
    SettingsStorage(BC::Key::widgetKey(BC::Key::TCW::key,cfg.headerKey())),
    d_config{cfg}
{
    const auto numChannels = cfg.numChannels();

    p_table = new QTableWidget(static_cast<int>(numChannels), 3, this);
    p_table->setHorizontalHeaderLabels({"Ch", "Name", "Enabled"});
    p_table->horizontalHeader()->setSectionResizeMode(LabelCol, QHeaderView::ResizeToContents);
    p_table->horizontalHeader()->setSectionResizeMode(NameCol, QHeaderView::Stretch);
    p_table->horizontalHeader()->setSectionResizeMode(EnabledCol, QHeaderView::ResizeToContents);
    p_table->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    p_table->verticalHeader()->setVisible(false);
    p_table->setSelectionMode(QAbstractItemView::NoSelection);
    p_table->setFocusPolicy(Qt::NoFocus);

    for(uint i=0; i<numChannels; ++i)
    {
        auto labelItem = new QTableWidgetItem(QString::number(i+1));
        labelItem->setFlags(Qt::ItemIsEnabled);
        labelItem->setTextAlignment(Qt::AlignCenter);
        p_table->setItem(static_cast<int>(i), LabelCol, labelItem);

        auto nameItem = new QTableWidgetItem(cfg.channelName(i));
        p_table->setItem(static_cast<int>(i), NameCol, nameItem);

        auto cb = new QCheckBox(this);
        connect(cb, &QCheckBox::toggled, this, [this,i](bool en){
            emit channelEnableChanged(d_config.headerKey(), i, en);
        });
        centerCellWidget(p_table, static_cast<int>(i), EnabledCol, cb);

        d_channelWidgets.push_back({cb});
    }

    connect(p_table, &QTableWidget::itemChanged, this, [this](QTableWidgetItem *item){
        if(!item || item->column() != NameCol)
            return;
        emit channelNameChanged(d_config.headerKey(),
                                static_cast<uint>(item->row()),
                                item->text());
    });

    auto vbl = new QVBoxLayout(this);
    vbl->setContentsMargins(0,0,0,0);
    vbl->addWidget(p_table);

    setFromConfig(cfg);
}

TemperatureControlWidget::~TemperatureControlWidget()
{
}

TemperatureControllerConfig &TemperatureControlWidget::toConfig()
{
    for(uint i=0; i<d_config.numChannels(); i++)
    {
        if((std::size_t)i < d_channelWidgets.size())
        {
            auto item = p_table->item(static_cast<int>(i), NameCol);
            d_config.setName(i, item ? item->text() : QString());
            d_config.setEnabled(i, d_channelWidgets.at(i).checkBox->isChecked());
        }
    }

    return d_config;
}

void TemperatureControlWidget::setFromConfig(const TemperatureControllerConfig &cfg)
{
    if(cfg.headerKey() != d_config.headerKey())
        return;

    for(uint i=0; i<cfg.numChannels(); ++i)
    {
        if((std::size_t)i < d_channelWidgets.size())
        {
            p_table->blockSignals(true);
            if(auto item = p_table->item(static_cast<int>(i), NameCol))
                item->setText(cfg.channelName(i));
            p_table->blockSignals(false);

            setChannelEnabled(cfg.headerKey(), i, cfg.channelEnabled(i));
        }
    }

    d_config = cfg;
}

void TemperatureControlWidget::setChannelEnabled(const QString key, uint ch, bool en)
{
    if(key != d_config.headerKey())
        return;

    if(ch >= d_channelWidgets.size())
        return;

    auto cb = d_channelWidgets[ch].checkBox;
    cb->blockSignals(true);
    cb->setChecked(en);
    cb->blockSignals(false);
}
