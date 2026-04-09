#include "pythonhardwarecontrolwidget.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QDesktopServices>
#include <QUrl>
#include <QGroupBox>

#include <hardware/core/hardwaremanager.h>
#include <hardware/core/hardwareprofilemanager.h>
#include <data/bcglobals.h>

PythonHardwareControlWidget::PythonHardwareControlWidget(const QString &hwKey, HardwareManager *hwm, QWidget *parent)
    : QWidget(parent), d_hwKey(hwKey), p_hwm(hwm)
{
    auto ki = BC::Key::parseKey(hwKey);
    auto scriptPath = HardwareProfileManager::instance().getPythonScriptPath(ki.first, ki.second);

    auto vbl = new QVBoxLayout(this);

    auto gb = new QGroupBox(QStringLiteral("Python Script"));
    auto gbLayout = new QVBoxLayout(gb);

    auto pathLabel = new QLabel(scriptPath.isEmpty() ? QStringLiteral("(no script configured)") : scriptPath);
    pathLabel->setWordWrap(true);
    gbLayout->addWidget(pathLabel);

    auto buttonLayout = new QHBoxLayout;

    auto openButton = new QPushButton(QStringLiteral("Open in Editor"));
    connect(openButton, &QPushButton::clicked, [scriptPath]() {
        if (!scriptPath.isEmpty())
            QDesktopServices::openUrl(QUrl::fromLocalFile(scriptPath));
    });
    openButton->setEnabled(!scriptPath.isEmpty());
    buttonLayout->addWidget(openButton);

    auto reloadButton = new QPushButton(QStringLiteral("Reload Script"));
    connect(reloadButton, &QPushButton::clicked, [this]() {
        p_statusLabel->setText(QStringLiteral("Reloading..."));
        QMetaObject::invokeMethod(p_hwm, [this, key=d_hwKey]() {
            p_hwm->reloadPythonScript(key);
        });
    });
    reloadButton->setEnabled(!scriptPath.isEmpty());
    buttonLayout->addWidget(reloadButton);

    gbLayout->addLayout(buttonLayout);

    p_statusLabel = new QLabel(QStringLiteral("Running"));
    gbLayout->addWidget(p_statusLabel);

    vbl->addWidget(gb);

    connect(p_hwm, &HardwareManager::pythonScriptReloadResult,
            this, &PythonHardwareControlWidget::onReloadResult);
}

void PythonHardwareControlWidget::onReloadResult(const QString &hwKey, bool success, const QString &msg)
{
    if (hwKey != d_hwKey)
        return;

    if (success)
        p_statusLabel->setText(QStringLiteral("Running"));
    else
        p_statusLabel->setText(QStringLiteral("Error: ") + msg);
}
