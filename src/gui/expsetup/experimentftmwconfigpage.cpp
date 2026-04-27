#include "experimentftmwconfigpage.h"

#include <QAbstractButton>
#include <QInputDialog>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>

#include <data/bcglobals.h>
#include <data/experiment/ftmwconfig.h>
#include <data/loadout/loadoutmanager.h>
#include <data/settings/hardwarekeys.h>
#include <hardware/core/runtimehardwareconfig.h>
#include <hardware/core/ftmwdigitizer/ftmwscope.h>
#include <hardware/optional/chirpsource/awg.h>

#include <gui/widget/chirpconfigwidget.h>
#include <gui/widget/ftmwconfigwidget.h>
#include <gui/widget/ftmwdigitizerconfigwidget.h>
#include <gui/widget/rfconfigwidget.h>

using namespace BC::Key::WizFtmw;

ExperimentFtmwConfigPage::ExperimentFtmwConfigPage(
        Experiment *exp,
        const QHash<RfConfig::ClockType, RfConfig::ClockFreq> &clocks,
        QWidget *parent)
    : ExperimentConfigPage(key, title, exp, parent)
{
    QString awgHwKey, digiHwKey;
    const auto &hw = RuntimeHardwareConfig::constInstance().getCurrentHardware();
    for (const auto &[k, v] : hw) {
        auto [type, label] = BC::Key::parseKey(k);
        if (type == BC::Key::AWG::key)
            awgHwKey = k;
        else if (type == RuntimeHardwareConfig::hardwareTypeOf<FtmwScope>())
            digiHwKey = k;
    }

    p_widget = new FtmwConfigWidget(awgHwKey, digiHwKey, clocks, false, this);
    connect(p_widget, &FtmwConfigWidget::edited, this, &ExperimentFtmwConfigPage::presetChanged);

    auto *layout = new QVBoxLayout(this);
    layout->addWidget(p_widget, 1);
    setLayout(layout);

    if (exp->d_number > 0 && exp->ftmwEnabled())
    {
        p_widget->initializeFromExperiment(*exp->ftmwConfig());
        p_widget->clearDirty();
    }
}

RfConfigWidget *ExperimentFtmwConfigPage::rfConfigWidget() const
{
    return p_widget->rfConfigWidget();
}

void ExperimentFtmwConfigPage::initialize()
{
}

bool ExperimentFtmwConfigPage::validate()
{
    if (!isEnabled() || !p_exp->ftmwConfig())
        return true;

    p_widget->clearTabErrors();
    bool out = true;

    // RF validation
    auto *rfWidget = p_widget->rfConfigWidget();
    auto ftmwType = p_exp->ftmwConfig()->d_type;

    if (ftmwType == FtmwConfig::LO_Scan) {
        if (rfWidget->getHwKey(RfConfig::UpLO).isEmpty()) {
            emit error("No upconversion LO set for LO Scan."_L1);
            p_widget->setTabError(0, true);
            out = false;
        }
        if (!rfWidget->commonLO() && rfWidget->getHwKey(RfConfig::DownLO).isEmpty()) {
            emit error("No downconversion LO set for LO Scan."_L1);
            p_widget->setTabError(0, true);
            out = false;
        }
    } else if (ftmwType == FtmwConfig::DR_Scan) {
        if (rfWidget->getHwKey(RfConfig::DRClock).isEmpty()) {
            emit error("No DR clock set for DR Scan."_L1);
            p_widget->setTabError(0, true);
            out = false;
        }
    } else {
        if (rfWidget->getHwKey(RfConfig::UpLO).isEmpty())
            emit warning("No upconversion LO set; assuming 0 MHz."_L1);
        if (!rfWidget->commonLO() && rfWidget->getHwKey(RfConfig::DownLO).isEmpty())
            emit warning("No downconversion LO set; assuming 0 MHz."_L1);
    }

    // Chirp validation
    auto *chirpWidget = p_widget->chirpConfigWidget();
    const auto chirpList = chirpWidget->getChirps().chirpList();

    if (chirpList.isEmpty()) {
        emit error("No chirp configured."_L1);
        p_widget->setTabError(1, true);
        out = false;
    } else {
        for (int i = 0; i < chirpList.size(); i++) {
            if (chirpList.at(i).isEmpty()) {
                emit error(QString("Chirp %1 is not configured").arg(i + 1));
                p_widget->setTabError(1, true);
                out = false;
            }
        }
    }

    // Protection marker warnings
    int mCount = 0;
    auto awgKeys = RuntimeHardwareConfig::constInstance().getActiveKeys<AWG>();
    if (!awgKeys.isEmpty()) {
        SettingsStorage awg(awgKeys.first(), SettingsStorage::Hardware);
        mCount = awg.get(BC::Key::AWG::markerCount, 0);
    }
    if (mCount > 0) {
        const auto &cc = chirpWidget->getChirps();
        const auto *prot = cc.findEnabledMarkerByRole(MarkerRole::Protection);
        const auto *gate = cc.findEnabledMarkerByRole(MarkerRole::Gate);
        if (!prot) {
            bool hasProt = false;
            for (const auto &m : cc.markerChannels()) {
                if (m.role == MarkerRole::Protection) { hasProt = true; break; }
            }
            if (!hasProt)
                emit warning("No protection marker is configured."_L1);
            else
                emit warning("Protection marker is disabled while the chirp is active."_L1);
        } else {
            if (prot->startTime >= 0.0)
                emit warning("Protection pulse starts at or after the chirp."_L1);
            if (prot->endTime <= 0.0)
                emit warning("Protection pulse ends at or before the chirp."_L1);
            if (gate) {
                if (prot->startTime > gate->startTime)
                    emit warning("Protection pulse starts after the amp enable pulse."_L1);
                if (prot->endTime < gate->endTime)
                    emit warning("Protection pulse ends before the amp enable pulse."_L1);
            }
        }
    }

    // Digitizer validation
    auto *digiWidget = p_widget->digiWidget();
    int numChirps = p_exp->ftmwConfig()->d_rfConfig.d_chirpConfig.numChirps();
    bool ba = digiWidget->blockAverageChecked();
    bool mr = digiWidget->multiRecordChecked();

    if (numChirps > 1) {
        if (!ba && !mr)
            emit warning("Number of chirps is >1, but digitizer is not configured for multiple records or block averaging."_L1);
        if (ba && digiWidget->numAverages() != numChirps)
            emit warning("Number of chirps does not match number of block averages."_L1);
    }
    if (mr && digiWidget->numRecords() != numChirps)
        emit warning("Number of chirps does not match number of digitizer records."_L1);

    if (digiWidget->numAnalogChecked() < 1) {
        emit error("No FID channel selected."_L1);
        p_widget->setTabError(2, true);
        out = false;
    } else if (digiWidget->numAnalogChecked() > 1) {
        emit error("Only 1 FID channel may be selected."_L1);
        p_widget->setTabError(2, true);
        out = false;
    }

    return out;
}

void ExperimentFtmwConfigPage::apply()
{
    if (!isEnabled() || !p_exp->ftmwEnabled())
        return;

    auto *cfg = p_exp->ftmwConfig();
    p_widget->rfConfigWidget()->toRfConfig(cfg->d_rfConfig);
    cfg->d_rfConfig.setChirpConfig(p_widget->chirpConfigWidget()->getChirps());
    p_widget->digiWidget()->toConfig(cfg->scopeConfig());
    if (!cfg->scopeConfig().d_analogChannels.empty())
        cfg->scopeConfig().d_fidChannel = cfg->scopeConfig().d_analogChannels.cbegin()->first;
}

void ExperimentFtmwConfigPage::commitFtmwPreset()
{
    using namespace Qt::StringLiterals;
    if (!isEnabled() || !p_widget->isDirty())
        return;

    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    if (activeName.isEmpty())
        return;

    const auto currentPresetName = LoadoutManager::instance().currentFtmwPresetName(activeName);
    const bool canOverwrite = !currentPresetName.isEmpty()
        && currentPresetName != BC::Store::LM::lastUsedFtmwPresetName;

    QMessageBox msgBox(this);
    msgBox.setWindowTitle(u"Save FTMW changes?"_s);
    msgBox.setText(u"The FTMW configuration has unsaved changes."_s);

    const QString overwriteLabel = canOverwrite
        ? QString(u"Overwrite \"%1\""_s).arg(currentPresetName)
        : u"Overwrite current preset"_s;
    QAbstractButton *overwriteBtn = static_cast<QAbstractButton*>(
        msgBox.addButton(overwriteLabel, QMessageBox::AcceptRole));
    overwriteBtn->setEnabled(canOverwrite);
    QAbstractButton *saveAsBtn = static_cast<QAbstractButton*>(
        msgBox.addButton(u"Save as new preset..."_s, QMessageBox::ActionRole));
    msgBox.addButton(u"Proceed without saving"_s, QMessageBox::DestructiveRole);

    msgBox.exec();
    auto *clicked = msgBox.clickedButton();
    const auto preset = p_widget->toFtmwPreset();

    if (clicked == overwriteBtn)
    {
        LoadoutManager::instance().putFtmwPreset(activeName, currentPresetName, preset);
        LoadoutManager::instance().putFtmwPreset(
            activeName, BC::Store::LM::lastUsedFtmwPresetName, preset);
        p_widget->clearDirty();
    }
    else if (clicked == saveAsBtn)
    {
        bool ok;
        auto name = QInputDialog::getText(
            this, u"Save FTMW Preset As"_s, u"Preset name:"_s,
            QLineEdit::Normal, {}, &ok).trimmed();

        bool saved = false;
        if (ok && !name.isEmpty() && name != BC::Store::LM::lastUsedFtmwPresetName)
        {
            bool doSave = true;
            if (LoadoutManager::instance().ftmwPresetExists(activeName, name))
            {
                const auto r = QMessageBox::question(
                    this, u"Overwrite Preset"_s,
                    QString(u"Preset \"%1\" already exists. Overwrite?"_s).arg(name),
                    QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
                doSave = (r == QMessageBox::Yes);
            }
            if (doSave)
            {
                LoadoutManager::instance().putFtmwPreset(activeName, name, preset);
                LoadoutManager::instance().putFtmwPreset(
                    activeName, BC::Store::LM::lastUsedFtmwPresetName, preset);
                LoadoutManager::instance().setCurrentFtmwPresetName(activeName, name);
                saved = true;
            }
        }

        if (!saved)
        {
            // Sub-dialog cancelled, invalid name, or overwrite declined — proceed without saving
            LoadoutManager::instance().putFtmwPreset(
                activeName, BC::Store::LM::lastUsedFtmwPresetName, preset);
            LoadoutManager::instance().setCurrentFtmwPresetName(
                activeName, BC::Store::LM::lastUsedFtmwPresetName);
        }
        p_widget->clearDirty();
    }
    else
    {
        // Proceed without saving
        LoadoutManager::instance().putFtmwPreset(
            activeName, BC::Store::LM::lastUsedFtmwPresetName, preset);
        LoadoutManager::instance().setCurrentFtmwPresetName(
            activeName, BC::Store::LM::lastUsedFtmwPresetName);
        p_widget->clearDirty();
    }
}
