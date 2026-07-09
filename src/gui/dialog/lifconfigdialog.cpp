#include "lifconfigdialog.h"

#include <QAbstractButton>
#include <QDialogButtonBox>
#include <QInputDialog>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>

#include <data/loadout/loadoutmanager.h>

#include <gui/lif/gui/lifconfigwidget.h>

using namespace Qt::StringLiterals;

LifConfigDialog::LifConfigDialog(const QString &digitizerHwKey, const QString &laserHwKey,
                                 QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle("LIF Configuration");
    resize(900, 700);

    auto *layout = new QVBoxLayout(this);

    p_widget = new LifConfigWidget(digitizerHwKey, laserHwKey, true, this);
    layout->addWidget(p_widget, 1);

    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    layout->addWidget(buttonBox);

    connect(buttonBox, &QDialogButtonBox::accepted, this, &LifConfigDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

void LifConfigDialog::accept()
{
    const auto activeName = LoadoutManager::instance().currentLoadoutName();

    if (!p_widget->isDirty()) {
        if (!activeName.isEmpty()) {
            LoadoutManager::instance().putLifPreset(
                activeName, BC::Store::LM::lastUsedLifPresetName, p_widget->toLifPreset());
        }
        QDialog::accept();
        return;
    }

    // Three-way prompt when dirty
    const auto currentPresetName = activeName.isEmpty()
        ? QString()
        : LoadoutManager::instance().currentLifPresetName(activeName);
    const bool canOverwrite = !currentPresetName.isEmpty()
        && currentPresetName != BC::Store::LM::lastUsedLifPresetName;

    QMessageBox msgBox(this);
    msgBox.setWindowTitle("Save LIF changes?"_L1);
    msgBox.setText("The LIF configuration has unsaved changes."_L1);

    const QString overwriteLabel = canOverwrite
        ? QString("Overwrite \"%1\"").arg(currentPresetName)
        : QString("Overwrite current preset");
    QAbstractButton *overwriteBtn = msgBox.addButton(overwriteLabel, QMessageBox::AcceptRole);
    overwriteBtn->setEnabled(canOverwrite);

    QAbstractButton *saveAsBtn  = msgBox.addButton("Save as new preset..."_L1, QMessageBox::ActionRole);
    QAbstractButton *proceedBtn = msgBox.addButton("Proceed without saving"_L1, QMessageBox::DestructiveRole);
    msgBox.addButton(QMessageBox::Cancel);

    msgBox.exec();
    auto *clicked = msgBox.clickedButton();

    if (clicked == overwriteBtn) {
        const auto preset = p_widget->toLifPreset();
        LoadoutManager::instance().putLifPreset(activeName, currentPresetName, preset);
        LoadoutManager::instance().putLifPreset(
            activeName, BC::Store::LM::lastUsedLifPresetName, preset);
        p_widget->clearDirty();
        QDialog::accept();
    } else if (clicked == saveAsBtn) {
        bool ok;
        auto name = QInputDialog::getText(
            this, "Save LIF Preset As"_L1, "Preset name:"_L1,
            QLineEdit::Normal, {}, &ok).trimmed();
        if (!ok || name.isEmpty())
            return;

        if (name == BC::Store::LM::lastUsedLifPresetName) {
            QMessageBox::warning(this, "Invalid Name"_L1, "That preset name is reserved."_L1);
            return;
        }

        if (LoadoutManager::instance().lifPresetExists(activeName, name)) {
            const auto r = QMessageBox::question(
                this, "Overwrite Preset"_L1,
                QString("Preset \"%1\" already exists. Overwrite?").arg(name),
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
            if (r != QMessageBox::Yes)
                return;
        }

        const auto preset = p_widget->toLifPreset();
        LoadoutManager::instance().putLifPreset(activeName, name, preset);
        LoadoutManager::instance().putLifPreset(
            activeName, BC::Store::LM::lastUsedLifPresetName, preset);
        LoadoutManager::instance().setCurrentLifPresetName(activeName, name);
        p_widget->clearDirty();
        QDialog::accept();
    } else if (clicked == proceedBtn) {
        if (!activeName.isEmpty()) {
            LoadoutManager::instance().putLifPreset(
                activeName, BC::Store::LM::lastUsedLifPresetName, p_widget->toLifPreset());
            LoadoutManager::instance().setCurrentLifPresetName(
                activeName, BC::Store::LM::lastUsedLifPresetName);
        }
        p_widget->clearDirty();
        QDialog::accept();
    }
    // else Cancel — return without accepting
}
