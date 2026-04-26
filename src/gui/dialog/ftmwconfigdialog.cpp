#include "ftmwconfigdialog.h"

#include <QAbstractButton>
#include <QDialogButtonBox>
#include <QInputDialog>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>

#include <data/loadout/hardwareloadout.h>
#include <data/loadout/loadoutmanager.h>

#include <gui/widget/ftmwconfigwidget.h>

using namespace Qt::StringLiterals;

FtmwConfigDialog::FtmwConfigDialog(const QString &awgHwKey, const QString &digiHwKey,
                                   const QHash<RfConfig::ClockType, RfConfig::ClockFreq> &currentClocks,
                                   QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle("FTMW Configuration");
    resize(900, 700);

    auto *layout = new QVBoxLayout(this);

    p_widget = new FtmwConfigWidget(awgHwKey, digiHwKey, currentClocks, true, this);
    layout->addWidget(p_widget, 1);

    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    layout->addWidget(buttonBox);

    connect(p_widget, &FtmwConfigWidget::applyClocks, this, &FtmwConfigDialog::applyClocks);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &FtmwConfigDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

void FtmwConfigDialog::accept()
{
    const auto activeName = LoadoutManager::instance().currentLoadoutName();

    if (!p_widget->isDirty()) {
        if (!activeName.isEmpty()) {
            LoadoutManager::instance().putFtmwPreset(
                activeName, BC::Store::LM::lastUsedFtmwPresetName, p_widget->toFtmwPreset());
        }
        QDialog::accept();
        return;
    }

    // Three-way prompt when dirty
    const auto currentPresetName = activeName.isEmpty()
        ? QString()
        : LoadoutManager::instance().currentFtmwPresetName(activeName);
    const bool canOverwrite = !currentPresetName.isEmpty()
        && currentPresetName != BC::Store::LM::lastUsedFtmwPresetName;

    QMessageBox msgBox(this);
    msgBox.setWindowTitle("Save FTMW changes?"_L1);
    msgBox.setText("The FTMW configuration has unsaved changes."_L1);

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
        const auto preset = p_widget->toFtmwPreset();
        LoadoutManager::instance().putFtmwPreset(activeName, currentPresetName, preset);
        LoadoutManager::instance().putFtmwPreset(
            activeName, BC::Store::LM::lastUsedFtmwPresetName, preset);
        p_widget->clearDirty();
        QDialog::accept();
    } else if (clicked == saveAsBtn) {
        bool ok;
        auto name = QInputDialog::getText(
            this, "Save FTMW Preset As"_L1, "Preset name:"_L1,
            QLineEdit::Normal, {}, &ok).trimmed();
        if (!ok || name.isEmpty())
            return;

        if (name == BC::Store::LM::lastUsedFtmwPresetName) {
            QMessageBox::warning(this, "Invalid Name"_L1, "That preset name is reserved."_L1);
            return;
        }

        if (LoadoutManager::instance().ftmwPresetExists(activeName, name)) {
            const auto r = QMessageBox::question(
                this, "Overwrite Preset"_L1,
                QString("Preset \"%1\" already exists. Overwrite?").arg(name),
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
            if (r != QMessageBox::Yes)
                return;
        }

        const auto preset = p_widget->toFtmwPreset();
        LoadoutManager::instance().putFtmwPreset(activeName, name, preset);
        LoadoutManager::instance().putFtmwPreset(
            activeName, BC::Store::LM::lastUsedFtmwPresetName, preset);
        LoadoutManager::instance().setCurrentFtmwPresetName(activeName, name);
        p_widget->clearDirty();
        QDialog::accept();
    } else if (clicked == proceedBtn) {
        if (!activeName.isEmpty()) {
            LoadoutManager::instance().putFtmwPreset(
                activeName, BC::Store::LM::lastUsedFtmwPresetName, p_widget->toFtmwPreset());
            LoadoutManager::instance().setCurrentFtmwPresetName(
                activeName, BC::Store::LM::lastUsedFtmwPresetName);
        }
        p_widget->clearDirty();
        QDialog::accept();
    }
    // else Cancel — return without accepting
}
