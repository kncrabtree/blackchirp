#include "clockdisplaybox.h"

#include <QGridLayout>
#include <QLabel>
#include <QMetaEnum>
#include <QToolButton>

#include <gui/style/themecolors.h>
#include <gui/util/numericformat.h>

using namespace Qt::Literals::StringLiterals;

ClockDisplayBox::ClockDisplayBox(QWidget *parent) :
    HardwareStatusBox(QString{}, parent)
{
    auto gl = new QGridLayout;
    gl->setSpacing(3);
    gl->setContentsMargins(3, 3, 3, 3);

    auto ct = QMetaEnum::fromType<RfConfig::ClockType>();

    for (int i = 0; i < ct.keyCount(); i++) {
        auto *nameLabel = new QLabel(QString(ct.key(i)), body());
        nameLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        nameLabel->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Preferred);

        auto *valueLabel = new QLabel(body());
        valueLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        valueLabel->setText(BC::Gui::formatNumberForDisplay(0.0, d_decimals) + " MHz"_L1);

        auto *cogButton = new QToolButton(body());
        cogButton->setIcon(ThemeColors::createThemedIcon(":/icons/cog-6-tooth.svg"_L1, ThemeColors::IconSecondary, body()));
        cogButton->setAutoRaise(true);
        cogButton->setIconSize({16, 16});

        gl->addWidget(nameLabel, i, 0);
        gl->addWidget(valueLabel, i, 1);
        gl->addWidget(cogButton, i, 2);

        auto type = static_cast<RfConfig::ClockType>(ct.value(i));
        ClockRow row;
        row.nameLabel = nameLabel;
        row.valueLabel = valueLabel;
        row.cogButton = cogButton;
        d_rows.insert(type, row);

        nameLabel->hide();
        valueLabel->hide();
        cogButton->hide();
    }

    gl->setColumnStretch(0, 0);
    gl->setColumnStretch(1, 1);
    gl->setColumnStretch(2, 0);
    body()->setLayout(gl);

    setTitle("Clock Configuration"_L1);
}

void ClockDisplayBox::updateFrequency(RfConfig::ClockType t, double f)
{
    auto it = d_rows.find(t);
    if (it == d_rows.end())
        return;
    it->valueLabel->setText(BC::Gui::formatNumberForDisplay(f, d_decimals) + " MHz"_L1);
}

void ClockDisplayBox::setClockHardware(RfConfig::ClockType type, const QString &hwKey, int output)
{
    auto it = d_rows.find(type);
    if (it == d_rows.end())
        return;

    it->hwKey = hwKey;

    if (hwKey.isEmpty()) {
        it->nameLabel->hide();
        it->valueLabel->hide();
        it->cogButton->hide();
        return;
    }

    auto tooltip = QString("%1 output %2"_L1).arg(hwKey).arg(output);
    it->nameLabel->setToolTip(tooltip);
    it->valueLabel->setToolTip(tooltip);

    disconnect(it->cogButton, &QToolButton::clicked, nullptr, nullptr);
    connect(it->cogButton, &QToolButton::clicked, this, [this, hwKey]() {
        emit clockHardwareRequested(hwKey);
    });

    it->nameLabel->show();
    it->valueLabel->show();
    it->cogButton->show();
}
