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
            title = QString("%1: %2"_L1).arg(parts[0], parts[1]);
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

    auto *configButton = new QToolButton(this);
    configButton->setIcon(ThemeColors::createThemedIcon(":/icons/cog-6-tooth.svg"_L1, ThemeColors::IconSecondary, this));
    configButton->setAutoRaise(true);
    configButton->setIconSize({16, 16});
    connect(configButton, &QToolButton::clicked, this, &HardwareStatusBox::configureRequested);
    titleRow->addWidget(configButton);

    outerLayout->addLayout(titleRow);

    p_body = new QWidget(this);
    outerLayout->addWidget(p_body);

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
    p_titleLabel->setText(t);
}
