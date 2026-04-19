#include "hardwarestatusbox.h"

#include <gui/style/themecolors.h>

#include <QHBoxLayout>
#include <QLabel>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidget>

using namespace Qt::Literals::StringLiterals;

HardwareStatusBox::HardwareStatusBox(const QString &key, QWidget *parent) :
    QFrame(parent), d_key{key}
{
    QString title;
    if (!d_key.isEmpty()) {
        auto parts = d_key.split('.');
        if (parts.size() >= 2)
            title = u"%1: %2"_s.arg(parts[0], parts[1]);
        else
            title = d_key;
    }

    auto *outerLayout = new QVBoxLayout(this);
    outerLayout->setContentsMargins(2, 2, 2, 2);
    outerLayout->setSpacing(2);

    auto *titleRow = new QHBoxLayout;
    titleRow->setContentsMargins(0, 0, 0, 0);
    titleRow->setSpacing(4);

    auto *collapseButton = new QToolButton(this);
    collapseButton->setIcon(ThemeColors::createThemedIcon(":/icons/chevron-down.svg"_L1, ThemeColors::IconSecondary, this));
    collapseButton->setAutoRaise(true);
    collapseButton->setIconSize({16, 16});
    titleRow->addWidget(collapseButton);

    p_titleLabel = new QLabel(title, this);
    auto f = p_titleLabel->font();
    f.setWeight(QFont::Bold);
    p_titleLabel->setFont(f);
    titleRow->addWidget(p_titleLabel);
    titleRow->addStretch();

    p_configButton = new QToolButton(this);
    p_configButton->setIcon(ThemeColors::createThemedIcon(":/icons/cog-6-tooth.svg"_L1, ThemeColors::IconSecondary, this));
    p_configButton->setAutoRaise(true);
    p_configButton->setIconSize({16, 16});
    connect(p_configButton, &QToolButton::clicked, this, &HardwareStatusBox::configureRequested);
    p_configButton->setToolTip(u"Open %1 Settings Dialog"_s.arg(d_key));
    titleRow->addWidget(p_configButton);

    outerLayout->addLayout(titleRow);

    p_body = new QWidget(this);
    outerLayout->addWidget(p_body);

    auto *separator = new QFrame(this);
    separator->setFrameShape(QFrame::HLine);
    separator->setFrameShadow(QFrame::Plain);
    separator->setLineWidth(1);
    separator->setStyleSheet(u"QFrame { color: %1; }"_s
                             .arg(ThemeColors::getCSSColor(ThemeColors::SubtleText, this)));
    outerLayout->addWidget(separator);

    connect(collapseButton, &QToolButton::clicked, this, [this, collapseButton]() {
        bool visible = !p_body->isVisible();
        p_body->setVisible(visible);
        collapseButton->setIcon(ThemeColors::createThemedIcon(
            visible ? ":/icons/chevron-down.svg"_L1 : ":/icons/chevron-right.svg"_L1,
            ThemeColors::IconSecondary, this));
    });

}

QWidget *HardwareStatusBox::body() const
{
    return p_body;
}

QSize HardwareStatusBox::sizeHint() const
{
    return {250, 1};
}

void HardwareStatusBox::setTitle(const QString &t)
{
    if(p_titleLabel)
        p_titleLabel->setText(t);
}

void HardwareStatusBox::setConfigButtonTooltip(const QString &t)
{
    if(p_configButton)
        p_configButton->setToolTip(t);
}
