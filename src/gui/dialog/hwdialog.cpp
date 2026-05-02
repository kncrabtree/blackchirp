#include "hwdialog.h"
#include <data/settings/hardwarekeys.h>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QTabWidget>

#include <gui/widget/hwsettingswidget.h>
#include <gui/style/themecolors.h>
#include <data/storage/settingsstorage.h>
#include <data/bcglobals.h>
#include <QToolTip>

HWDialog::HWDialog(const QString &key, QWidget *controlWidget, QWidget *managedWidget, QWidget *parent)
    : QDialog(parent), d_hwKey(key), p_controlWidget(controlWidget),
      p_managedWidget(managedWidget ? managedWidget : controlWidget)
{
    setAttribute(Qt::WA_DeleteOnClose);
    setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
    setWindowIcon(ThemeColors::createThemedIcon(":/icons/bc_logo_trans.svg", ThemeColors::IconPrimary, this));

    auto vbl = new QVBoxLayout;

    SettingsStorage s(key, SettingsStorage::Hardware);
    // Storage key format is "hwType.label"; the registry uses only the hwType component.
    auto hwType = key.split(BC::Key::hwIndexSep).first();
    auto impl = s.get(BC::Key::HW::model, key);
    setWindowTitle(QString("%1 Settings").arg(key));

    // Model info row
    auto nl = new QHBoxLayout;
    nl->addWidget(new QLabel("Model"), 0);
    nl->addWidget(new QLabel(impl, this), 1);
    vbl->addLayout(nl);

    auto tabWidget = new QTabWidget(this);

    if (controlWidget) {
        auto controlContainer = new QWidget;
        auto cvbl = new QVBoxLayout(controlContainer);
        auto cLabel = new QLabel("Changes made in this section will be applied immediately.");
        cLabel->setWordWrap(true);
        cLabel->setAlignment(Qt::AlignCenter);
        cvbl->addWidget(cLabel, 0);
        cvbl->addWidget(controlWidget);
        cvbl->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Minimum,QSizePolicy::Expanding));
        tabWidget->addTab(controlContainer, "Control");
    }

    auto settingsContainer = new QWidget;
    auto svbl = new QVBoxLayout(settingsContainer);
    auto sLabel = new QLabel("Changes made in this section will only be applied when this dialog is closed with the Ok button. Editing these settings incorrectly may result in unexpected behavior. Consider backing up your config file before making changes.");
    sLabel->setWordWrap(true);
    sLabel->setAlignment(Qt::AlignCenter);
    svbl->addWidget(sLabel, 0);

    p_settingsWidget = new HwSettingsWidget(hwType, impl, HwSettingsMode::Edit, key, this);
    svbl->addWidget(p_settingsWidget, 1);
    tabWidget->addTab(settingsContainer, "Settings");

    vbl->addWidget(tabWidget, 1);

    auto extraRow = new QHBoxLayout;

    p_testButton = new QPushButton("Test Connection"_L1, this);
    p_testButton->setIcon(ThemeColors::createThemedIcon(":/icons/check-circle.svg"_L1, ThemeColors::IconPrimary, this));
    connect(p_testButton, &QPushButton::clicked, this, [this](){
        d_testInFlight = true;
        p_testButton->setEnabled(false);
        p_testButton->setToolTip("Test already in progress"_L1);
        p_statusLabel->setText("Testing..."_L1);
        p_statusLabel->setStyleSheet(QString("QLabel { color: %1; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::StatusWarning, this)));
        emit requestTestConnection(d_hwKey);
    });
    extraRow->addWidget(p_testButton);

    p_statusLabel = new QLabel(this);
    extraRow->addWidget(p_statusLabel, 1);

    auto commButton = new QPushButton("Communication Settings..."_L1, this);
    connect(commButton, &QPushButton::clicked, this, [this](){
        emit requestCommunicationDialog(d_hwKey);
    });
    extraRow->addWidget(commButton);

    vbl->addLayout(extraRow, 0);

    auto bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Close);
    connect(bb->button(QDialogButtonBox::Ok), &QPushButton::clicked, this, &HWDialog::accept);
    connect(bb->button(QDialogButtonBox::Close), &QPushButton::clicked, this, &HWDialog::reject);

    vbl->addWidget(bb, 0);
    setLayout(vbl);
}

void HWDialog::discardControlWidget()
{
    if (p_controlWidget) {
        auto ss = dynamic_cast<SettingsStorage*>(p_controlWidget);
        if (ss)
            ss->discardChanges();
    }
}

void HWDialog::accept()
{
    p_settingsWidget->saveToStorage(d_hwKey);
    QDialog::accept();
}

QSize HWDialog::sizeHint() const
{
    return {600, 550};
}

void HWDialog::setControlWidgetEnabled(bool enabled)
{
    if (p_managedWidget)
        p_managedWidget->setEnabled(enabled);
}

void HWDialog::setConnectionStatus(bool connected)
{
    if (d_testInFlight)
        return;

    if (connected) {
        p_statusLabel->setText("Connected"_L1);
        p_statusLabel->setStyleSheet(QString("QLabel { color: %1; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::StatusSuccess, this)));
    } else {
        p_statusLabel->setText("Disconnected"_L1);
        p_statusLabel->setStyleSheet(QString("QLabel { color: %1; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::StatusError, this)));
    }
    p_statusLabel->setToolTip(""_L1);
}

void HWDialog::onConnectionResult(const QString &hwKey, bool success, const QString &msg)
{
    if (hwKey != d_hwKey)
        return;

    d_testInFlight = false;
    p_testButton->setEnabled(true);
    p_testButton->setToolTip(""_L1);

    if (success) {
        p_statusLabel->setText("Connected"_L1);
        p_statusLabel->setStyleSheet(QString("QLabel { color: %1; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::StatusSuccess, this)));
        p_statusLabel->setToolTip(""_L1);
    } else {
        p_statusLabel->setText("Connection failed"_L1);
        p_statusLabel->setStyleSheet(QString("QLabel { color: %1; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::StatusError, this)));
        p_statusLabel->setToolTip(msg);
    }
}
